'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts';

// ─── Types ───────────────────────────────────────────────────────
type Action = 'allow' | 'stepup' | 'block';
type Stats = {
  processed: number; precision: number; recall: number;
  fp_per_1k_good: number; rupees_saved: number; margin: number;
};
type RiskEvent = {
  txn_id: number; amount: number; action: Action;
  reason_codes: string[]; true_fraud: number; risk_score: number;
};
type AblationStage = {
  stage: string; PR_AUC_raw: number; rupees_per_1k: number;
};
type Ring = {
  ring_id: number; severity: string; score: number;
  n_accounts: number; n_fraud: number; fraud_rate: number;
  n_devices: number; device_concentration: number;
  total_amount_inr: number; n_emails: number; n_bins: number;
  shared_devices: string[]; shared_emails: string[];
};

const inr = (v: number) => '₹' + Math.round(v).toLocaleString('en-IN');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getRupeeData(metrics: any) {
  if (!metrics?.rupees) return [];
  return [
    { name: 'Model', value: Math.round(metrics.rupees.model), fill: '#22c55e' },
    { name: 'Step-up All', value: Math.round(metrics.rupees.stepup_all), fill: '#eab308' },
    { name: 'Rule: Top 3%', value: Math.round(metrics.rupees.rule_amount), fill: '#8b8d98' },
    { name: 'Allow All', value: Math.round(metrics.rupees.allow_all), fill: '#ef4444' },
    { name: 'Block All', value: Math.round(metrics.rupees.block_all), fill: '#5c5e6a' },
  ];
}

// ─── Component ───────────────────────────────────────────────────
export default function Page() {
  const [tab, setTab] = useState<string>('shield');
  const [stats, setStats] = useState<Stats>({ processed: 0, precision: 0, recall: 0, fp_per_1k_good: 0, rupees_saved: 0, margin: 0.2 });
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [margin, setMargin] = useState(20);
  const [thresholds, setThresholds] = useState({ stepup: 0.005, block: 0.302 });
  const [mix, setMix] = useState({ allow: 0, stepup: 0, block: 0 });
  const [cm, setCm] = useState({ tp: 0, fp: 0, fn: 0, tn: 0 });
  const [ablation, setAblation] = useState<AblationStage[]>([]);
  const [rings, setRings] = useState<Ring[]>([]);
  const [offlineMetrics, setOfflineMetrics] = useState<any>({});
  const total = useRef(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Load static data on mount
  useEffect(() => {
    fetch('/api/ablation').then(r => r.json()).then(d => setAblation(d.stages || [])).catch(() => {});
    fetch('/api/rings').then(r => r.json()).then(d => setRings(d.rings || [])).catch(() => {});
    fetch('/api/offline-metrics').then(r => r.json()).then(d => setOfflineMetrics(d)).catch(() => {});
    fetch('/api/margin?margin=0.2', { method: 'POST' }).then(r => r.json())
      .then(d => setThresholds({ stepup: d.thresholds?.stepup ?? 0, block: d.thresholds?.block ?? 1 })).catch(() => {});
  }, []);

  // Replay step
  const step = useCallback(async () => {
    const r = await fetch('/api/simulate/step?n=1', { method: 'POST' });
    const d = await r.json();
    if (!d.events?.length) { setRunning(false); return; }
    const e: RiskEvent = d.events[0];
    setEvents((prev: RiskEvent[]) => [e, ...prev].slice(0, 40));
    total.current++;
    setMix(prev => ({ ...prev, [e.action]: prev[e.action] + 1 }));
    const flagged = e.action !== 'allow';
    const fraud = !!e.true_fraud;
    setCm(prev => ({
      tp: prev.tp + (flagged && fraud ? 1 : 0),
      fp: prev.fp + (flagged && !fraud ? 1 : 0),
      fn: prev.fn + (!flagged && fraud ? 1 : 0),
      tn: prev.tn + (!flagged && !fraud ? 1 : 0),
    }));
    setStats(d.stats);
  }, []);

  // Timer
  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(step, speed);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, speed, step]);

  // Margin slider
  const handleMargin = async (val: number) => {
    setMargin(val);
    const r = await fetch(`/api/margin?margin=${val / 100}`, { method: 'POST' });
    const d = await r.json();
    setThresholds({ stepup: d.thresholds?.stepup ?? 0, block: d.thresholds?.block ?? 1 });
  };

  const totalMix = total.current || 1;
  const mixPct = { allow: (mix.allow / totalMix) * 100, stepup: (mix.stepup / totalMix) * 100, block: (mix.block / totalMix) * 100 };

  // Pie chart data for action mix
  const pieData = [
    { name: 'Allow', value: mix.allow, color: 'var(--green)' },
    { name: 'Step-up', value: mix.stepup, color: 'var(--amber)' },
    { name: 'Block', value: mix.block, color: 'var(--red)' },
  ].filter(function(d) { return d.value > 0; });

  // Rupee comparison data for results tab
  const rupeeData = getRupeeData(offlineMetrics);

  return (
    <div className="shell">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">R</div>
          <div>
            <div className="header-title">RiskShield</div>
            <div className="header-subtitle">AI Risk Manager · Track 02</div>
          </div>
        </div>
        <div className="header-right">
          <select className="select" value={speed} onChange={e => setSpeed(Number(e.target.value))}>
            <option value={800}>Slow</option>
            <option value={300}>Normal</option>
            <option value={80}>Fast</option>
          </select>
          <button className={running ? 'btn' : 'btn btn-primary'} onClick={() => setRunning(p => !p)}>
            {running ? 'Pause' : 'Start Replay'}
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="tabs">
        <button className={`tab ${tab === 'shield' ? 'active' : ''}`} onClick={() => setTab('shield')}>Live Console</button>
        <button className={`tab ${tab === 'results' ? 'active' : ''}`} onClick={() => setTab('results')}>Results</button>
        <button className={`tab ${tab === 'rings' ? 'active' : ''}`} onClick={() => setTab('rings')}>Abuse Rings</button>
      </div>

      {/* ═══ LIVE CONSOLE TAB ═══ */}
      {tab === 'shield' && (
        <>
          {/* KPIs */}
          <div className="kpi-row">
            <div className="kpi-card hero">
              <div className="kpi-label">Rupees Saved</div>
              <div className="kpi-value">{inr(stats.rupees_saved)}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Processed</div>
              <div className="kpi-value">{stats.processed.toLocaleString('en-IN')}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Precision</div>
              <div className="kpi-value">{stats.precision ? stats.precision.toFixed(3) : '—'}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Recall</div>
              <div className="kpi-value">{stats.recall ? stats.recall.toFixed(3) : '—'}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">FP / 1k Good</div>
              <div className="kpi-value">{stats.fp_per_1k_good ? stats.fp_per_1k_good.toFixed(2) : '—'}</div>
            </div>
          </div>

          <div className="grid-2">
            {/* Left: Decision tape + bottom row */}
            <div>
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Decision Tape</span>
                  <span className="card-badge">{running ? 'Live' : events.length ? 'Paused' : 'Ready'}</span>
                </div>
                <div style={{ maxHeight: 420, overflowY: 'auto' }}>
                  {events.length === 0 ? (
                    <div className="empty">Press Start Replay to stream held-out transactions</div>
                  ) : events.map((e, i) => (
                    <div key={`${e.txn_id}-${i}`} className="tape-row">
                      <span className="txn-id">#{e.txn_id}</span>
                      <span className="txn-amt">{inr(e.amount)}</span>
                      <span className={`pill pill-${e.action}`}>{e.action}</span>
                      <span className="reason">{e.reason_codes.join(' · ')}</span>
                      <span className={`dot ${e.true_fraud ? 'dot-fraud' : 'dot-safe'}`} title={e.true_fraud ? 'Chargeback' : 'Clean'} />
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom: confusion matrix + action mix chart */}
              <div className="grid-2-equal">
                <div className="card">
                  <div className="card-header"><span className="card-title">Confusion Matrix</span></div>
                  <div className="card-body">
                    <div className="cm-grid">
                      <div className="cm-corner" />
                      <div className="cm-label">Flagged</div>
                      <div className="cm-label">Allowed</div>
                      <div className="cm-label">Fraud</div>
                      <div className="cm-cell cm-tp"><span className="cm-num">{cm.tp}</span><span className="cm-sub">TP</span></div>
                      <div className="cm-cell cm-fn"><span className="cm-num">{cm.fn}</span><span className="cm-sub">FN</span></div>
                      <div className="cm-label">Clean</div>
                      <div className="cm-cell cm-fp"><span className="cm-num">{cm.fp}</span><span className="cm-sub">FP</span></div>
                      <div className="cm-cell cm-tn"><span className="cm-num">{cm.tn}</span><span className="cm-sub">TN</span></div>
                    </div>
                  </div>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Action Distribution</span></div>
                  <div className="card-body">
                    {pieData.length > 0 ? (
                      <div className="chart-wrap-sm">
                        <ResponsiveContainer>
                          <PieChart>
                            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                              innerRadius={30} outerRadius={55} paddingAngle={3}>
                              {pieData.map((d, i) => <Cell key={i} fill={d.color} />)}
                            </Pie>
                            <Legend wrapperStyle={{ fontSize: 11, color: '#8b8d98' }} />
                            <Tooltip formatter={(v: number) => v.toLocaleString()} contentStyle={{ background: '#16181d', border: '1px solid #2a2d35', borderRadius: 6, fontSize: 12 }} />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="empty">Start replay to see distribution</div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Right sidebar */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {/* Margin slider */}
              <div className="card">
                <div className="card-header"><span className="card-title">Merchant Economics</span></div>
                <div className="slider-section">
                  <div className="slider-row">
                    <span>Contribution margin</span>
                    <strong>{margin}%</strong>
                  </div>
                  <input type="range" min={2} max={80} value={margin} onChange={e => handleMargin(Number(e.target.value))} />
                </div>
              </div>

              {/* Threshold band */}
              <div className="card">
                <div className="card-header"><span className="card-title">Score → Action</span></div>
                <div className="card-body">
                  <div className="band">
                    <div className="band-allow" style={{ width: `${Math.max(8, thresholds.stepup * 100)}%` }}>allow</div>
                    <div className="band-stepup" style={{ width: `${Math.max(8, (thresholds.block - thresholds.stepup) * 100)}%` }}>step-up</div>
                    <div className="band-block" style={{ width: `${Math.max(8, (1 - thresholds.block) * 100)}%` }}>block</div>
                  </div>
                  <div className="scale-row"><span>0.0</span><span>0.5</span><span>1.0</span></div>
                </div>
              </div>

              {/* Action mix bars */}
              <div className="card">
                <div className="card-header"><span className="card-title">Action Mix</span></div>
                <div className="card-body">
                  {(['allow', 'stepup', 'block'] as const).map(a => (
                    <div key={a} className="mix-row">
                      <span className={`mix-label mix-label-${a}`}>{a === 'stepup' ? 'step-up' : a}</span>
                      <div className="mix-track"><div className={`mix-fill mix-fill-${a}`} style={{ width: `${mixPct[a]}%` }} /></div>
                      <span className="mix-pct">{mixPct[a].toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="note" style={{ borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)' }}>
                Red dots are confirmed chargebacks from held-out labels, revealed <b>after</b> each decision. Drag the margin slider to see how a 5% electronics merchant gets different thresholds from a 60% SaaS merchant.
              </div>
            </div>
          </div>
        </>
      )}

      {/* ═══ RESULTS TAB ═══ */}
      {tab === 'results' && (
        <div>
          {/* Headline metrics */}
          <div className="kpi-row" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <div className="kpi-card hero">
              <div className="kpi-label">PR-AUC (Raw)</div>
              <div className="kpi-value">{offlineMetrics.pr_auc_raw?.toFixed(4) ?? '—'}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Precision</div>
              <div className="kpi-value">{offlineMetrics.precision?.toFixed(3) ?? '—'}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Recall</div>
              <div className="kpi-value">{offlineMetrics.recall?.toFixed(3) ?? '—'}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">95% CI</div>
              <div className="kpi-value" style={{ fontSize: 16 }}>
                {offlineMetrics.ci ? `[${offlineMetrics.ci[0].toFixed(3)}, ${offlineMetrics.ci[1].toFixed(3)}]` : '—'}
              </div>
            </div>
          </div>

          <div className="grid-2-equal">
            {/* Ablation chart */}
            <div className="card">
              <div className="card-header"><span className="card-title">Ablation Study — PR-AUC by Feature Stage</span></div>
              <div className="card-body">
                {ablation.length > 0 ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer>
                      <BarChart data={ablation} layout="vertical" margin={{ left: 100, right: 16, top: 4, bottom: 4 }}>
                        <XAxis type="number" domain={[0, 1]} tick={{ fill: '#5c5e6a', fontSize: 11 }} />
                        <YAxis type="category" dataKey="stage" tick={{ fill: '#8b8d98', fontSize: 11 }} width={95} />
                        <Tooltip contentStyle={{ background: '#16181d', border: '1px solid #2a2d35', borderRadius: 6, fontSize: 12 }}
                          formatter={(v: number) => v.toFixed(4)} />
                        <Bar dataKey="PR_AUC_raw" radius={[0, 4, 4, 0]}>
                          {ablation.map((_, i) => (
                            <Cell key={i} fill={i === ablation.length - 1 ? '#22c55e' : '#3a3d45'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : <div className="empty">Run pipeline to see ablation</div>}
              </div>
            </div>

            {/* Rupee comparison */}
            <div className="card">
              <div className="card-header"><span className="card-title">₹ Lost per 1,000 Transactions</span></div>
              <div className="card-body">
                {rupeeData.length > 0 ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer>
                      <BarChart data={rupeeData} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
                        <XAxis dataKey="name" tick={{ fill: '#8b8d98', fontSize: 10 }} />
                        <YAxis tick={{ fill: '#5c5e6a', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#16181d', border: '1px solid #2a2d35', borderRadius: 6, fontSize: 12 }}
                          formatter={(v: number) => `₹${v.toLocaleString('en-IN')}`} />
                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                          {rupeeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : <div className="empty">Run pipeline to see comparison</div>}
              </div>
            </div>
          </div>

          {/* Return risk + per-merchant thresholds */}
          <div className="grid-2-equal" style={{ marginTop: 12 }}>
            <div className="card">
              <div className="card-header"><span className="card-title">Return-Risk Scorer</span></div>
              <div className="card-body">
                {offlineMetrics.return_metrics ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                    <div><div className="kpi-label">PR-AUC</div><div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--mono)', color: 'var(--green)' }}>{offlineMetrics.return_metrics.pr_auc?.toFixed(4)}</div></div>
                    <div><div className="kpi-label">Abuse Rate</div><div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--mono)' }}>{(offlineMetrics.return_metrics.abuse_rate * 100).toFixed(1)}%</div></div>
                    <div><div className="kpi-label">Avg Cost/Return</div><div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--mono)', color: 'var(--amber)' }}>₹{offlineMetrics.return_metrics.avg_abuse_cost_inr}</div></div>
                  </div>
                ) : <div className="empty">No return data</div>}
              </div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Per-Merchant Thresholds</span></div>
              <div className="card-body">
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ color: 'var(--text-secondary)', textAlign: 'left' }}>
                      <th style={{ padding: '6px 0', fontWeight: 500 }}>Margin</th>
                      <th style={{ padding: '6px 0', fontWeight: 500 }}>Step-up ≥</th>
                      <th style={{ padding: '6px 0', fontWeight: 500 }}>Block ≥</th>
                    </tr>
                  </thead>
                  <tbody style={{ fontFamily: 'var(--mono)' }}>
                    {[{ m: '5%', su: '0.003', bl: '0.092' }, { m: '20%', su: '0.005', bl: '0.302' }, { m: '60%', su: '0.014', bl: '0.570' }].map(r => (
                      <tr key={r.m} style={{ borderTop: '1px solid var(--border)' }}>
                        <td style={{ padding: '8px 0' }}>{r.m}</td>
                        <td style={{ padding: '8px 0', color: 'var(--amber)' }}>{r.su}</td>
                        <td style={{ padding: '8px 0', color: 'var(--red)' }}>{r.bl}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Drift + honest weaknesses */}
          <div className="card" style={{ marginTop: 12 }}>
            <div className="card-header"><span className="card-title">Known Weaknesses (Honest Reporting)</span></div>
            <div className="card-body" style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--text-secondary)' }}>
              <p><b style={{ color: 'var(--text)' }}>Mid-value transactions (Q3):</b> PR-AUC 0.819 — too large for card-testing signature, too small for amount-anomaly features.</p>
              <p style={{ marginTop: 8 }}><b style={{ color: 'var(--text)' }}>New accounts:</b> 9.5 FP per 1,000 vs 0.5 for returning — thin history causes over-flagging.</p>
              <p style={{ marginTop: 8 }}><b style={{ color: 'var(--text)' }}>Adversarial drift AUC:</b> {offlineMetrics.drift_auc?.toFixed(3) ?? '—'} — distributions genuinely differ between train/test. Dropped features: {offlineMetrics.dropped?.join(', ') || 'none'}.</p>
            </div>
          </div>
        </div>
      )}

      {/* ═══ RINGS TAB ═══ */}
      {tab === 'rings' && (
        <div>
          {rings.length === 0 ? (
            <div className="empty" style={{ padding: 60 }}>No abuse rings detected above threshold</div>
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
                  {ring.shared_devices?.length > 0 && (
                    <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-dim)' }}>
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
  );
}
