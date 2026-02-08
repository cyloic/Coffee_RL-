"""
Coffee Supply Chain RL - Streamlit Dashboard
Professional dashboard for coffee lending policy comparison and simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime

# Must be first Streamlit command
st.set_page_config(
    page_title="Coffee RL Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern, professional styling
st.markdown("""
<style>
    :root {
        --primary-color: #1e3a5f;
        --accent-color: #d97706;
        --success-color: #059669;
        --warning-color: #f59e0b;
        --danger-color: #dc2626;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    .subheader-text {
        font-size: 1.3rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #d97706;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .metric-card h3 {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        color: #1e3a5f;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #6b7280;
        padding: 0.75rem 1.5rem !important;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #d97706 !important;
        border-bottom-color: #d97706 !important;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .content-box {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-ready {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-pending {
        background: #fef3c7;
        color: #92400e;
    }
    
    .sidebar-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Try to import RL dependencies
try:
    from stable_baselines3 import PPO
    from coffee_env import CoffeeLendingEnv
    from baseline_policies import (
        RandomPolicy, RuleBasedPolicy, LogisticRegressionPolicy,
        evaluate_policy
    )
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    st.error(f"⚠️ RL dependencies not available: {e}")


@st.cache_data
def load_dataset(path='data/processed/coffee_loans_500.csv'):
    """Load and cache dataset"""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found: {path}")
        return None


@st.cache_resource
def load_rl_model(model_path):
    """Load and cache RL model"""
    if not RL_AVAILABLE:
        return None
    try:
        model = PPO.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_baseline_results(path='results/baselines'):
    """Load baseline comparison results"""
    try:
        result_files = list(Path(path).glob('baseline_comparison_*.json'))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                return json.load(f)
        return None
    except Exception as e:
        st.warning(f"No baseline results found: {e}")
        return None


def create_comparison_chart(results_data):
    """Create interactive comparison chart"""
    if not results_data or 'results' not in results_data:
        return None
    
    policies = list(results_data['results'].keys())
    returns = [results_data['results'][p]['mean_return'] * 100 for p in policies]
    defaults = [results_data['results'][p]['mean_defaults'] * 100 for p in policies]
    
    fig = go.Figure()
    
    # Returns bars
    fig.add_trace(go.Bar(
        name='Portfolio Return (%)',
        x=policies,
        y=returns,
        marker_color='lightblue',
        text=[f'{r:.1f}%' for r in returns],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Policy Performance Comparison',
        xaxis_title='Policy',
        yaxis_title='Return (%)',
        height=500,
        showlegend=True
    )
    
    return fig


def create_risk_return_scatter(results_data):
    """Create risk vs return scatter plot"""
    if not results_data or 'results' not in results_data:
        return None
    
    policies = []
    returns = []
    defaults = []
    sharpe = []
    
    for name, res in results_data['results'].items():
        policies.append(name)
        returns.append(res['mean_return'] * 100)
        defaults.append(res['mean_defaults'] * 100)
        sharpe.append(res.get('sharpe_ratio', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=defaults,
        y=returns,
        mode='markers+text',
        marker=dict(
            size=[s * 10 + 10 for s in sharpe],
            color=returns,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Return (%)")
        ),
        text=policies,
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>Default: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Risk vs Return Analysis',
        xaxis_title='Default Rate (%)',
        yaxis_title='Portfolio Return (%)',
        height=500
    )
    
    return fig


def simulate_loan_decision(model, mill_data, env):
    """Simulate a loan decision for given mill"""
    # Normalize features like the environment does
    state = np.array([
        (mill_data['years_operating'] - 2) / (14 - 2),
        mill_data['utilization_rate'],
        float(mill_data['has_certification']),
        (mill_data['num_farmers'] - 150) / (900 - 150),
        (mill_data['coffee_price'] - 123.5) / (413.6 - 123.5),
        (mill_data['rainfall_mm'] - 11.75) / (150.46 - 11.75),
        (mill_data['capacity_tons'] - 30) / (150 - 30),
        (mill_data['debt_to_revenue'] - 0) / (40 - 0),
        (mill_data['buyer_contracts'] - 1) / (6 - 1),
        1.0,  # Full capital
        0.0   # No defaults yet
    ], dtype=np.float32)
    
    # Get action from model
    action, _ = model.predict(state, deterministic=True)
    action = int(action)
    
    loan_amounts = {0: 0, 1: 50_000, 2: 150_000, 3: 300_000}
    decision_map = {0: "Reject", 1: "Approve: $50k", 2: "Approve: $150k", 3: "Approve: $300k"}
    decision = decision_map[action]
    amount = loan_amounts[action]
    
    # Calculate risk score (simple heuristic)
    risk_score = (
        (1 - state[0]) * 0.25 +  # Experience
        (1 - state[1]) * 0.20 +  # Utilization
        (1 - state[2]) * 0.20 +  # Certification
        (1 - state[4]) * 0.25 +  # Coffee price
        state[7] * 0.10          # Debt ratio
    )
    
    return {
        'decision': decision,
        'action': action,
        'amount': amount,
        'risk_score': risk_score,
        'confidence': 1 - risk_score
    }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-logo"><h1 style="color: #1e3a5f; margin: 0;">COFFEE RL</h1></div>', unsafe_allow_html=True)
        st.markdown("Supply Chain Optimization Engine")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Policy Comparison", "Loan Simulator", 
             "Training Analytics", "Dataset Explorer"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model selector
        model_files = list(Path('models').glob('*.zip')) if Path('models').exists() else []
        if model_files:
            selected_model = st.selectbox(
                "Select RL Model",
                model_files,
                format_func=lambda x: x.stem
            )
        else:
            selected_model = None
            st.info("No trained models available. Train a model first.")
        
        st.markdown("---")
        st.caption("Version 1.0.0 | Coffee Supply Chain RL")
        st.caption("Last updated: January 2026")
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if page == "Dashboard":
        st.markdown('<p class="main-header">Coffee Supply Chain Lending</p>', unsafe_allow_html=True)
        st.markdown('<p class="subheader-text">Reinforcement Learning for Agricultural Finance in Rwanda</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Load latest results
        baseline_results = load_baseline_results()
        
        # Key Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Policies Tested",
                value="7",
                delta="RL + 6 Baselines"
            )
        
        with col2:
            if baseline_results and 'results' in baseline_results:
                best_return = max([r['mean_return'] * 100 for r in baseline_results['results'].values()])
                st.metric(
                    label="Best Return",
                    value=f"{best_return:.1f}%",
                    delta="Portfolio Return"
                )
            else:
                st.metric(label="Best Return", value="N/A")
        
        with col3:
            df = load_dataset()
            if df is not None:
                st.metric(
                    label="Dataset Size",
                    value=f"{len(df):,}",
                    delta="Loans"
                )
            else:
                st.metric(label="Dataset Size", value="N/A")
        
        with col4:
            if baseline_results:
                st.markdown('<span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
                st.metric(
                    label="Model Status",
                    value="",
                    delta="Trained & Evaluated"
                )
            else:
                st.markdown('<span class="status-badge status-pending">Pending</span>', unsafe_allow_html=True)
                st.metric(
                    label="Model Status",
                    value="",
                    delta="Run training"
                )
        
        st.markdown("---")
        
        # Project Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
            st.markdown("""
            This project uses **Reinforcement Learning** to optimize lending decisions 
            for coffee washing stations in Rwanda. The RL agent learns to:
            
            - **Maximize Returns**: Earn interest on repaid loans
            - **Manage Risk**: Minimize default rates
            - **Social Impact**: Help more coffee farmers
            - **Balance Trade-offs**: Profit vs. Risk vs. Impact
            """)
            
            st.markdown('<h3 style="color: #1e3a5f; margin-top: 1.5rem;">Technology Stack</h3>', unsafe_allow_html=True)
            tech_col1, tech_col2 = st.columns(2)
            with tech_col1:
                st.markdown("""
                - **Algorithm**: Proximal Policy Optimization (PPO)
                - **Environment**: Custom Gymnasium environment
                """)
            with tech_col2:
                st.markdown("""
                - **Dataset**: 5000 hybrid loans
                - **Baselines**: Random, Rule-Based, Logistic Regression
                """)
        
        with col2:
            st.markdown('<h2 class="section-header">Quick Stats</h2>', unsafe_allow_html=True)
            if df is not None:
                st.metric("Total Loans", f"{len(df):,}")
                st.metric("Default Rate", f"{df['default'].mean():.1%}")
                st.metric("Avg Loan Size", f"${df['loan_amount_usd'].mean():,.0f}")
                st.metric("Coffee Mills", f"{df['mill_id'].nunique()}")
        
        # Additional Section
        st.markdown("---")
        st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)
        
        step_col1, step_col2, step_col3 = st.columns(3)
        
        with step_col1:
            st.markdown("""
            ### 1. Environment
            The agent observes coffee mill characteristics and market conditions
            """)
        
        with step_col2:
            st.markdown("""
            ### 2. Decision
            The RL agent decides loan amount: Reject, 50k, 150k, or 300k USD
            """)
        
        with step_col3:
            st.markdown("""
            ### 3. Outcome
            Loan repayment or default determines reward signal for learning
            """)
    
    # ========================================================================
    # PAGE: POLICY COMPARISON
    # ========================================================================
    elif page == "Policy Comparison":
        st.markdown('<p class="main-header">Policy Comparison</p>', unsafe_allow_html=True)
        st.markdown("Compare RL agent against baseline policies to identify the best lending strategy")
        
        baseline_results = load_baseline_results()
        
        if baseline_results is None:
            st.error("No baseline results found. Run: python baseline_policies.py")
            return
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Performance Overview", "Risk Analysis", "Detailed Metrics"])
        
        with tab1:
            st.markdown('<h2 class="section-header">Return Comparison</h2>', unsafe_allow_html=True)
            fig = create_comparison_chart(baseline_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<h2 class="section-header">Risk vs Return Analysis</h2>', unsafe_allow_html=True)
            fig2 = create_risk_return_scatter(baseline_results)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.markdown('<h2 class="section-header">Detailed Metrics</h2>', unsafe_allow_html=True)
            
            results = baseline_results['results']
            metrics_df = pd.DataFrame({
                'Policy': list(results.keys()),
                'Return (%)': [r['mean_return'] * 100 for r in results.values()],
                'Default (%)': [r['mean_defaults'] * 100 for r in results.values()],
                'Sharpe Ratio': [r.get('sharpe_ratio', 0) for r in results.values()],
                'Farmers Helped': [int(r['mean_farmers']) for r in results.values()]
            })
            
            metrics_df = metrics_df.sort_values('Return (%)', ascending=False)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Winner announcement
        st.markdown("---")
        best_policy = max(results.items(), key=lambda x: x[1]['mean_return'])
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success(f"**Best Performing Policy**: {best_policy[0]}\n\n**Portfolio Return**: {best_policy[1]['mean_return']*100:.1f}%")
    
    # ========================================================================
    # PAGE: LIVE SIMULATOR
    # ========================================================================
    elif page == "Loan Simulator":
        st.markdown('<p class="main-header">Loan Decision Simulator</p>', unsafe_allow_html=True)
        st.markdown("Input mill characteristics and see real-time RL agent decisions")
        
        if not RL_AVAILABLE:
            st.error("RL dependencies not available. Install stable-baselines3")
            return
        
        if selected_model is None:
            st.warning("No model selected. Train a model first.")
            return
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_rl_model(str(selected_model))
            env = CoffeeLendingEnv('data/processed/coffee_loans_500.csv')
        
        if model is None:
            st.error("Failed to load model")
            return
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h2 class="section-header">Mill Characteristics</h2>', unsafe_allow_html=True)
            
            years_operating = st.slider("Years Operating", 2, 14, 7)
            utilization_rate = st.slider("Utilization Rate", 0.0, 1.0, 0.5, 0.05)
            has_certification = st.checkbox("Has Fair Trade Certification", value=True)
            num_farmers = st.slider("Number of Farmers", 150, 900, 400, 50)
            coffee_price = st.slider("Coffee Price (USD/lb)", 1.0, 4.0, 2.0, 0.1)
            rainfall_mm = st.slider("Rainfall (mm/year)", 20, 150, 80, 10)
            capacity_tons = st.slider("Capacity (tons/year)", 30, 150, 64, 10)
            debt_to_revenue = st.slider("Debt-to-Revenue Ratio", 0.0, 0.4, 0.15, 0.05)
            buyer_contracts = st.slider("Buyer Contracts", 1, 6, 3)
        
        with col2:
            st.markdown('<h2 class="section-header">Agent Decision</h2>', unsafe_allow_html=True)
            
            mill_data = {
                'years_operating': years_operating,
                'utilization_rate': utilization_rate,
                'has_certification': has_certification,
                'num_farmers': num_farmers,
                'coffee_price': coffee_price,
                'rainfall_mm': rainfall_mm,
                'capacity_tons': capacity_tons,
                'debt_to_revenue': debt_to_revenue,
                'buyer_contracts': buyer_contracts
            }
            
            result = simulate_loan_decision(model, mill_data, env)
            
            # Decision display
            st.markdown(f"### {result['decision']}")
            
            if result['amount'] > 0:
                st.success(f"**Loan Amount:** ${result['amount']:,}")
            else:
                st.error("**Loan Rejected**")
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Score", f"{result['risk_score']:.1%}")
            with col_b:
                st.metric("Confidence", f"{result['confidence']:.1%}")
        
        st.markdown("---")
        st.markdown('<h2 class="section-header">Risk Factor Breakdown</h2>', unsafe_allow_html=True)
        
        risk_factors = {
            'Experience': (1 - (years_operating - 2) / 12) * 25,
            'Utilization': (1 - utilization_rate) * 20,
            'Certification': (0 if has_certification else 1) * 20,
            'Coffee Price': (1 - (coffee_price - 1) / 3) * 25,
            'Debt Ratio': debt_to_revenue * 10 / 0.4
        }
        
        risk_df = pd.DataFrame({
            'Factor': list(risk_factors.keys()),
            'Risk (%)': list(risk_factors.values())
        })
        
        fig = px.bar(risk_df, x='Factor', y='Risk (%)', 
                    color='Risk (%)', color_continuous_scale='RdYlGn_r',
                    title="Risk Assessment by Factor")
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE: TRAINING ANALYTICS
    # ========================================================================
    elif page == "Training Analytics":
        st.markdown('<p class="main-header">Training Analytics</p>', unsafe_allow_html=True)
        st.markdown("View training progress and performance metrics from the latest training run")
        
        # Find latest training run
        results_dirs = list(Path('results').glob('ppo_baseline_*')) if Path('results').exists() else []
        
        if not results_dirs:
            st.warning("No training runs found. Train a model first.")
            return
        
        latest_run = max(results_dirs, key=lambda p: p.stat().st_mtime)
        
        st.info(f"Latest Run: {latest_run.name}")
        st.markdown("---")
        
        # Tabs for different analytics
        tab1, tab2 = st.tabs(["Training Progress", "Evaluation Results"])
        
        with tab1:
            st.markdown('<h2 class="section-header">Episode Rewards Over Time</h2>', unsafe_allow_html=True)
            
            log_file = latest_run / 'training_log.json'
            if log_file.exists():
                with open(log_file) as f:
                    log_data = json.load(f)
                
                if log_data['episode_rewards']:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=log_data['episode_rewards'],
                        mode='lines',
                        name='Episode Reward',
                        line=dict(color='#d97706', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(217, 119, 6, 0.1)'
                    ))
                    fig.update_layout(
                        title='Reward Progression During Training',
                        xaxis_title='Episode',
                        yaxis_title='Total Episode Reward',
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training log found for this run")
        
        with tab2:
            st.markdown('<h2 class="section-header">Evaluation Metrics</h2>', unsafe_allow_html=True)
            
            eval_file = latest_run / 'evaluation_results.json'
            if eval_file.exists():
                with open(eval_file) as f:
                    eval_data = json.load(f)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Reward", f"{eval_data['mean_reward']:,.0f}")
                with col2:
                    st.metric("Portfolio Return", f"{eval_data['mean_portfolio_return']:.1%}")
                with col3:
                    st.metric("Default Rate", f"{eval_data['mean_default_rate']:.1%}")
                with col4:
                    st.metric("Farmers Helped", f"{eval_data['mean_farmers_helped']:,.0f}")
                
                st.markdown("---")
                st.markdown('<h3 style="color: #1e3a5f;">Full Evaluation Report</h3>', unsafe_allow_html=True)
                st.json(eval_data)
            else:
                st.info("No evaluation results found for this run")
        
        # TensorBoard link
        st.markdown("---")
        st.info(f"For detailed TensorBoard metrics, run:\n\n```bash\ntensorboard --logdir {latest_run / 'tensorboard'}\n```")
    
    # ========================================================================
    # PAGE: DATASET EXPLORER
    # ========================================================================
    elif page == "Dataset Explorer":
        st.markdown('<p class="main-header">Dataset Explorer</p>', unsafe_allow_html=True)
        st.markdown("Analyze and explore the coffee lending dataset")
        
        df = load_dataset()
        if df is None:
            st.error("Dataset not found")
            return
        
        st.markdown("---")
        
        # Summary stats in tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Data Table"])
        
        with tab1:
            st.markdown('<h2 class="section-header">Dataset Summary</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Loans", f"{len(df):,}")
            with col2:
                st.metric("Default Rate", f"{df['default'].mean():.1%}")
            with col3:
                st.metric("Avg Loan Size", f"${df['loan_amount_usd'].mean():,.0f}")
            with col4:
                st.metric("Coffee Mills", f"{df['mill_id'].nunique()}")
        
        with tab2:
            st.markdown('<h2 class="section-header">Data Filters</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                province_filter = st.multiselect(
                    "Province",
                    options=df['province'].unique(),
                    default=None
                )
            
            with col2:
                cert_filter = st.selectbox(
                    "Certification",
                    ["All", "Certified", "Not Certified"]
                )
            
            with col3:
                default_filter = st.selectbox(
                    "Default Status",
                    ["All", "Defaulted", "Repaid"]
                )
            
            # Apply filters
            filtered_df = df.copy()
            if province_filter:
                filtered_df = filtered_df[filtered_df['province'].isin(province_filter)]
            if cert_filter != "All":
                filtered_df = filtered_df[filtered_df['has_certification'] == (cert_filter == "Certified")]
            if default_filter != "All":
                filtered_df = filtered_df[filtered_df['default'] == (1 if default_filter == "Defaulted" else 0)]
            
            st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} loans**")
            
            # Visualizations
            st.markdown('<h3 style="color: #1e3a5f; margin-top: 1.5rem;">Distribution Charts</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(filtered_df, x='loan_amount_usd', nbins=30,
                                 title='Loan Amount Distribution',
                                 labels={'loan_amount_usd': 'Loan Amount (USD)', 'count': 'Count'})
                fig.update_traces(marker_color='#d97706')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(filtered_df, names='province', title='Loans by Province',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown('<h2 class="section-header">Raw Data</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                province_filter = st.multiselect(
                    "Province",
                    options=df['province'].unique(),
                    default=None,
                    key="province_tab3"
                )
            
            with col2:
                cert_filter = st.selectbox(
                    "Certification",
                    ["All", "Certified", "Not Certified"],
                    key="cert_tab3"
                )
            
            with col3:
                default_filter = st.selectbox(
                    "Default Status",
                    ["All", "Defaulted", "Repaid"],
                    key="default_tab3"
                )
            
            # Apply filters
            filtered_df = df.copy()
            if province_filter:
                filtered_df = filtered_df[filtered_df['province'].isin(province_filter)]
            if cert_filter != "All":
                filtered_df = filtered_df[filtered_df['has_certification'] == (cert_filter == "Certified")]
            if default_filter != "All":
                filtered_df = filtered_df[filtered_df['default'] == (1 if default_filter == "Defaulted" else 0)]
            
            st.dataframe(filtered_df.head(100), use_container_width=True)


if __name__ == '__main__':
    main()