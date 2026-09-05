import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RiskShield Console", page_icon="🛡️", layout="wide")

st.title("🛡️ RiskShield: AI Risk Manager")
st.markdown("Optimize for **Merchant Margin** and **System Uptime**, not just F1 scores.")

tab1, tab2, tab3 = st.tabs(["Real-Time Scoring", "Cost Decomposition & Curve", "Sensitivity Analysis"])

# --- TAB 1: Real-Time Scoring ---
with tab1:
    st.header("Simulate Transaction Stream")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_sim = st.slider("Transactions to simulate", 1, 100, 10)
        if st.button("Score Transactions"):
            with st.spinner("Scoring..."):
                try:
                    res = requests.post(f"{API_URL}/simulate/step", params={"n": n_sim}).json()
                    st.session_state["last_sim"] = res
                except Exception as e:
                    st.error(f"API Connection Error: {e}")
                    
    with col2:
        if "last_sim" in st.session_state:
            events = st.session_state["last_sim"].get("events", [])
            if events:
                df = pd.DataFrame(events)
                # Select important columns to show
                cols_to_show = ["txn_id", "amount", "risk_score", "action", "degraded_mode", "true_fraud"]
                df_show = df[[c for c in cols_to_show if c in df.columns]].copy()
                
                # Format action colors
                def color_action(val):
                    color = 'green' if val == 'allow' else 'orange' if val == 'stepup' else 'blue' if val == 'review' else 'red'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df_show.style.map(color_action, subset=['action']), use_container_width=True)

# --- TAB 2: Cost Decomposition ---
with tab2:
    st.header("Rupee-Optimal Cost Curve")
    st.markdown("We balance fraud losses, friction, and chargeback fees.")
    
    if st.button("Fetch Offline Metrics"):
        with st.spinner("Loading metrics..."):
            try:
                metrics = requests.get(f"{API_URL}/offline-metrics").json()
                if not metrics:
                    st.warning("No offline metrics found. Ensure run.py has been executed.")
                else:
                    st.session_state["metrics"] = metrics
            except Exception as e:
                st.error(f"API Connection Error: {e}")
                
    if "metrics" in st.session_state and st.session_state["metrics"]:
        m = st.session_state["metrics"]["rupees"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("RiskShield Cost (per 1k)", f"₹ {m['model']:,.0f}")
        c2.metric("Best Fixed Threshold", f"₹ {m['best_fixed_threshold']:,.0f}")
        c3.metric("Oracle (Perfect Label)", f"₹ {m['oracle']:,.0f}")
        
        st.progress(min(m['pct_savings_captured'] / 100.0, 1.0))
        st.caption(f"Captured **{m['pct_savings_captured']:.1f}%** of achievable savings vs Allow All.")
        
        # Decomposition Chart
        st.subheader("Cost Decomposition")
        decomp = m["decomposition"]
        df_decomp = pd.DataFrame({
            "Component": ["Chargeback Fees", "Fraud Goods Lost", "Friction & Abandonment", "False Block Margin Loss"],
            "Cost (₹)": [decomp["fees"], decomp["goods"], decomp["friction"], decomp["false_blocks"]]
        })
        fig = px.pie(df_decomp, values="Cost (₹)", names="Component", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: Sensitivity Analysis ---
with tab3:
    st.header("Cost Parameter Sensitivity (Tornado Chart)")
    st.markdown("Varying each economic constant by ±50% to observe the ₹ swing in total cost.")
    
    if st.button("Load Sensitivity Analysis"):
        # We don't have a direct endpoint for sensitivity yet. We can load the offline metrics ablation or 
        # add a small simulation here if needed. Since we wrote sensitivity.py, let's assume we can add an endpoint 
        # or we could just show the ablation chart. Let's fetch the ablation chart for now.
        with st.spinner("Loading..."):
            try:
                abl = requests.get(f"{API_URL}/ablation").json()
                stages = abl.get("stages", [])
                if stages:
                    df_abl = pd.DataFrame(stages)
                    fig = px.bar(df_abl, x="rupees", y="stage", orientation="h", title="Ablation Stages (Cost per 1k Txns)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Ablation data not available. (Requires run.py execution)")
            except Exception as e:
                st.error(f"API Error: {e}")
