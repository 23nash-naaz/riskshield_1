'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

type Action = 'allow' | 'stepup' | 'block';
type RiskEvent = {
  txn_id: number;
  amount: number;
  action: Action;
  reason_codes: string[];
  true_fraud: number;
  risk_score: number;
  card?: string;
  expected_cost_inr?: Record<string, number>;
  saved_vs_allow_inr?: number;
};
type Stats = {
  processed: number;
  precision: number;
  recall: number;
  fp_per_1k_good: number;
  rupees_saved: number;
};
type Ring = {
  ring_id: number; severity: string; score: number;
  n_accounts: number; n_fraud: number; fraud_rate: number;
  n_devices: number; device_concentration: number;
  total_amount_inr: number;
  shared_devices?: string[];
};

const inr = (v: number) => '₹' + Math.round(v).toLocaleString('en-IN');
const clock = () => new Date().toLocaleTimeString('en-IN', { hour12: false });
const cardMask = (e: RiskEvent) => {
  const n = String(e.txn_id).padStart(4, '0').slice(-4);
  return (e.card || 'card').toUpperCase().slice(0, 4) + ' •• ' + n;
};

export default function Page() {
  const [page, setPage] = useState('live');
  const [merchant, setMerchant] = useState(0.20);
  const [mid, setMid] = useState('MID: rzp_live_8kQ2…f4');
  const [speed, setSpeed] = useState(450);
  const [running, setRunning] = useState(false);
  const [filterQuery, setFilterQuery] = useState('');
  
  const [stats, setStats] = useState<Stats>({ processed: 0, precision: 0, recall: 0, fp_per_1k_good: 0, rupees_saved: 0 });
  const [events, setEvents] = useState<(RiskEvent & { time: string })[]>([]);
  const [disputes, setDisputes] = useState<(RiskEvent & { deadline: string })[]>([]);
  
  const [thresholds, setThresholds] = useState({ stepup: 0.3, block: 0.8 });
  const [rings, setRings] = useState<Ring[]>([]);
  const [health, setHealth] = useState<any>({});
  
  const [drawerEvent, setDrawerEvent] = useState<RiskEvent | null>(null);
  const [packText, setPackText] = useState('');
  
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const latencies = useRef<number[]>([]);
  const [p50, setP50] = useState(0);

  // Fetch initial data
  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(d => setHealth(d)).catch(() => {});
    fetch('/api/rings').then(r => r.json()).then(d => setRings(d.rings || [])).catch(() => {});
    updateMargin(0.20);
  }, []);

  const updateMargin = async (m: number) => {
    setMerchant(m);
    try {
      const r = await fetch('/api/margin?margin=' + m, { method: 'POST' });
      const d = await r.json();
      const t = d.thresholds || {};
      const su = t.stepup ?? t.block ?? 1;
      const bl = t.block ?? 1;
      setThresholds({ stepup: su, block: bl });
    } catch(e) {}
  };

  const handleMerchantSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const m = Number(e.target.value);
    setMid('MID: rzp_live_' + Math.random().toString(36).slice(2, 6) + '…' + Math.random().toString(36).slice(2, 4));
    updateMargin(m);
  };

  const step = useCallback(async () => {
    const t0 = performance.now();
    try {
      const r = await fetch('/api/simulate/step?n=1', { method: 'POST' });
      const ms = performance.now() - t0;
      latencies.current.push(ms);
      if (latencies.current.length > 200) latencies.current.shift();
      
      const d = await r.json();
      if (!d.events || !d.events.length) {
        setRunning(false);
        return;
      }
      
      const e = d.events[0];
      const newEvent = { ...e, time: clock() };
      
      setEvents(prev => [newEvent, ...prev].slice(0, 60));
      setStats(d.stats);
      
      if (e.true_fraud && e.action === 'allow') {
        const dl = new Date(Date.now() + 7 * 864e5).toLocaleDateString('en-IN');
        setDisputes(prev => [{ ...e, deadline: dl }, ...prev]);
      }
      
      const sorted = [...latencies.current].sort((a, b) => a - b);
      setP50(sorted[Math.floor(sorted.length / 2)] || 0);
    } catch(e) {
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(step, speed);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, speed, step]);

  const toggleRun = () => setRunning(prev => !prev);
  
  const generatePack = async (e: RiskEvent) => {
    try {
      const r = await fetch('/api/dispute-pack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          txn: { TransactionID: e.txn_id, TransactionAmt: e.amount, DeviceInfo: 1, TransactionDT: Date.now() / 1000, avs_match: true, cvv_match: true, ip_country: 'IN', bill_country: 'IN' },
          history: [], dispute_code: '10.4'
        })
      });
      const p = await r.json();
      setPackText(p.text);
    } catch(err) {
      setPackText("Failed to generate pack.");
    }
  };

  const openDrawer = (e: RiskEvent) => {
    setDrawerEvent(e);
    setPackText('');
  };
  
  const filteredEvents = useMemo(() => {
    if (!filterQuery) return events;
    const q = filterQuery.toLowerCase();
    return events.filter(e => 
      e.txn_id.toString().includes(q) || 
      e.amount.toString().includes(q) || 
      e.action.includes(q) || 
      e.reason_codes.join(' ').toLowerCase().includes(q)
    );
  }, [events, filterQuery]);

  const b1w = thresholds.stepup * 100;
  const b2w = (thresholds.block - thresholds.stepup) * 100;
  const b3w = (1 - thresholds.block) * 100;

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100%', overflow: 'hidden' }}>
      <aside>
        <div className="logo"><div className="mark">R</div><div><b>RiskShield</b><span>RISK OPERATIONS</span></div></div>
        <div className="merch">
          <label>Merchant</label>
          <select value={merchant} onChange={handleMerchantSelect}>
            <option value={0.05}>Veltron Electronics · 5% margin</option>
            <option value={0.20}>Kanchi Home &amp; Living · 20%</option>
            <option value={0.60}>Luma SaaS Pvt Ltd · 60%</option>
          </select>
          <div className="mline mono">{mid}</div>
        </div>
        <nav>
          <a className={page === 'live' ? 'on' : ''} onClick={() => setPage('live')}>▦ Live decisions</a>
          <a className={page === 'disputes' ? 'on' : ''} onClick={() => setPage('disputes')}>⚑ Disputes {disputes.length > 0 && <span className="badge">{disputes.length}</span>}</a>
          <a className={page === 'settings' ? 'on' : ''} onClick={() => setPage('settings')}>⚙ Cost model</a>
          <a className={page === 'rings' ? 'on' : ''} onClick={() => setPage('rings')}>⚯ Abuse rings</a>
        </nav>
        <div className="syshealth">
          <div><span>Scorer</span><span className="okdot">● healthy</span></div>
          <div><span>Feature store</span><span className="okdot">● warm</span></div>
          <div><span>Model</span><span className="mono">lgbm-iso v1.3</span></div>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <span className="env">● LIVE</span>
          <input type="text" placeholder="Filter tape: amount / reason / action…" value={filterQuery} onChange={e => setFilterQuery(e.target.value)} />
          <div className="spacer"></div>
          <select value={speed} onChange={e => setSpeed(Number(e.target.value))}>
            <option value={900}>Slow</option>
            <option value={450}>Normal</option>
            <option value={140}>Fast</option>
          </select>
          <button className="primary" onClick={toggleRun}>{running ? '⏸ Pause traffic' : '▶ Start traffic'}</button>
          <div className="avatar">RM</div>
        </div>

        <div className="content">
          {/* Live Page */}
          {page === 'live' && (
            <div className="page on">
              <div className="kpis">
                <div className="kpi"><label>₹ saved vs allow-all</label><div className="v" id="saved">{inr(stats.rupees_saved)}</div></div>
                <div className="kpi"><label>Processed</label><div className="v">{stats.processed.toLocaleString('en-IN')}</div></div>
                <div className="kpi"><label>Precision</label><div className="v">{stats.precision ? stats.precision.toFixed(3) : '—'}</div></div>
                <div className="kpi"><label>Recall</label><div className="v">{stats.recall ? stats.recall.toFixed(3) : '—'}</div></div>
                <div className="kpi"><label>FP / 1k good</label><div className="v">{stats.fp_per_1k_good ? stats.fp_per_1k_good.toFixed(2) : '—'}</div></div>
                <div className="kpi"><label>p50 latency</label><div className="v">{p50 ? p50.toFixed(0) + ' ms' : '—'}</div></div>
              </div>

              <div className="card">
                <h2>Decision tape — held-out future window
                  <span className="right"><span className="rc">{running ? 'streaming' : events.length ? 'paused' : 'idle'}</span></span>
                </h2>
                <table>
                  <thead><tr><th>Time</th><th>Txn</th><th>Card</th><th>Amount</th><th>Decision</th><th>Signals</th><th>CB</th></tr></thead>
                  <tbody>
                    {filteredEvents.map((e, idx) => (
                      <tr key={`${e.txn_id}-${idx}`} className="txn" onClick={() => openDrawer(e)}>
                        <td className="rc">{e.time}</td>
                        <td className="rc">#{e.txn_id}</td>
                        <td>{cardMask(e)}</td>
                        <td>{inr(e.amount)}</td>
                        <td><span className={`chip ${e.action}`}>{e.action.toUpperCase()}</span></td>
                        <td className="rc">{e.reason_codes.join(' · ')}</td>
                        <td><span className={`dot ${e.true_fraud ? 'fraud' : 'ok'}`}></span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Disputes Page */}
          {page === 'disputes' && (
            <div className="page on">
              <div className="card">
                <h2>Chargebacks on allowed transactions — representment queue</h2>
                <table>
                  <thead><tr><th>Txn</th><th>Amount</th><th>Reason code</th><th>Deadline</th><th></th></tr></thead>
                  <tbody>
                    {disputes.map((e, idx) => (
                      <tr key={`${e.txn_id}-${idx}`} className="txn">
                        <td className="rc">#{e.txn_id}</td>
                        <td>{inr(e.amount)}</td>
                        <td className="rc">10.4 · Fraud, card absent</td>
                        <td className="rc">{e.deadline}</td>
                        <td><button onClick={() => openDrawer(e)}>Evidence →</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {disputes.length === 0 && <div className="empty">No disputes yet. They appear when an allowed transaction turns out to be a confirmed chargeback.</div>}
              </div>
            </div>
          )}

          {/* Settings Page */}
          {page === 'settings' && (
            <div className="page on">
              <div className="card">
                <h2>Merchant economics — thresholds are solved from these numbers, not tuned</h2>
                <div style={{ padding: 16, maxWidth: 560 }}>
                  <div className="mono" style={{ marginBottom: 6 }}>Contribution margin <b style={{ color: 'var(--brass)' }}>{Math.round(merchant * 100)}%</b></div>
                  <input type="range" min={2} max={80} value={merchant * 100} onChange={e => updateMargin(Number(e.target.value) / 100)} />
                  <div className="slider-band">
                    <div className="b1" style={{ width: b1w + '%' }}>ALLOW</div>
                    <div className="b2" style={{ width: b2w + '%' }}>STEP-UP</div>
                    <div className="b3" style={{ width: b3w + '%' }}>BLOCK</div>
                  </div>
                  <div className="rc" style={{ marginTop: 8 }}>Risk score 0 → 1. Where each action becomes rupee-optimal for this merchant.</div>
                  <table style={{ marginTop: 18 }}>
                    <tbody>
                      <tr><td>Chargeback dispute fee</td><td className="mono" style={{ textAlign: 'right' }}>₹1,500</td></tr>
                      <tr><td>3DS abandonment (good users)</td><td className="mono" style={{ textAlign: 'right' }}>8%</td></tr>
                      <tr><td>3DS fraud-stop rate</td><td className="mono" style={{ textAlign: 'right' }}>90%</td></tr>
                      <tr><td>Step-up opex per call</td><td className="mono" style={{ textAlign: 'right' }}>₹2</td></tr>
                    </tbody>
                  </table>
                  <div className="rc" style={{ marginTop: 8 }}>These are the model's most important inputs. In production each merchant sets their own from real dispute history.</div>
                </div>
              </div>
            </div>
          )}
          
          {/* Rings Page */}
          {page === 'rings' && (
            <div className="page on">
              {rings.length === 0 ? (
                <div className="empty" style={{ paddingTop: 60 }}>No abuse rings detected above threshold. Run python run.py first.</div>
              ) : (
                <div className="rings-grid">
                  {rings.map(ring => (
                    <div key={ring.ring_id} className="ring-card">
                      <div className="ring-top">
                        <span className="ring-name">Ring #{ring.ring_id}</span>
                        <span className={`severity severity-${ring.severity.toLowerCase()}`}>{ring.severity}</span>
                      </div>
                      <div className="ring-stats">
                        <div className="ring-stat">Accounts<strong>{ring.n_accounts.toLocaleString()}</strong></div>
                        <div className="ring-stat">Fraud txns<strong>{ring.n_fraud}</strong></div>
                        <div className="ring-stat">Fraud rate<strong>{(ring.fraud_rate * 100).toFixed(1)}%</strong></div>
                        <div className="ring-stat">Devices<strong>{ring.n_devices.toLocaleString()}</strong></div>
                        <div className="ring-stat">Dev conc.<strong>{ring.device_concentration}×</strong></div>
                        <div className="ring-stat">Total ₹<strong>{inr(ring.total_amount_inr)}</strong></div>
                      </div>
                      {ring.shared_devices && ring.shared_devices.length > 0 && (
                        <div className="ring-shared">
                          Shared devices: {ring.shared_devices.slice(0, 3).join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="statusbar">
          <span>latency p50: {p50 ? p50.toFixed(0) + ' ms' : '—'}</span>
          <span>queue: {health.replay_remaining ? health.replay_remaining.toLocaleString('en-IN') : '—'}</span>
          <span>store: 9,267 accounts</span>
          <span>region: ap-south-1</span><span style={{ marginLeft: 'auto' }}>RiskShield v1.3 · defense-only</span>
        </div>
      </div>

      {/* Drawer */}
      <div className={`drawer ${drawerEvent ? 'open' : ''}`}>
        <div className="dh">
          <b>Txn {drawerEvent ? '#' + drawerEvent.txn_id : ''}</b>
          <div className="spacer"></div>
          <button onClick={() => setDrawerEvent(null)}>✕</button>
        </div>
        <div className="body">
          {drawerEvent && (
            <>
              <div className="kv"><span>Amount</span><b className="mono">{inr(drawerEvent.amount)}</b></div>
              <div className="kv"><span>Card</span><span className="mono">{cardMask(drawerEvent)}</span></div>
              <div className="kv"><span>Risk score (calibrated)</span><b className="mono">{drawerEvent.risk_score}</b></div>
              <div className="kv"><span>Decision</span><span className={`chip ${drawerEvent.action}`}>{drawerEvent.action.toUpperCase()}</span></div>
              <div className="kv"><span>Outcome</span><span>{drawerEvent.true_fraud ? '⚠ confirmed chargeback' : 'clean (so far)'}</span></div>
              
              <div style={{ marginTop: 14, font: '600 10px Archivo', letterSpacing: '.1em', color: 'var(--dim)' }}>EXPECTED COST OF EACH ACTION</div>
              <div className="costrow">
                {(['allow', 'stepup', 'block'] as Action[]).map(a => (
                  <div key={a} className={`costbox ${a === drawerEvent.action ? 'best' : ''}`}>
                    <label>{a}</label>
                    <div className="cv">{drawerEvent.expected_cost_inr?.[a] != null ? inr(drawerEvent.expected_cost_inr[a]) : '—'}</div>
                  </div>
                ))}
              </div>
              <div className="rc">System picked <b>{drawerEvent.action}</b> — the cheapest expected loss. Saved {inr(drawerEvent.saved_vs_allow_inr || 0)} vs allowing blindly.</div>
              
              <div style={{ marginTop: 14, font: '600 10px Archivo', letterSpacing: '.1em', color: 'var(--dim)' }}>SIGNALS (coarse by design)</div>
              <div className="rc" style={{ marginTop: 6 }} dangerouslySetInnerHTML={{ __html: drawerEvent.reason_codes.join('<br>') }} />
              
              {drawerEvent.true_fraud > 0 && (
                <>
                  <button className="primary" style={{ marginTop: 16 }} onClick={() => generatePack(drawerEvent)}>Generate representment pack</button>
                  {packText && <pre style={{ marginTop: 10 }}>{packText}</pre>}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
