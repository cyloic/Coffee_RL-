import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coffee RL Dashboard", layout="wide")

st.title("Coffee RL — Project Dashboard")
st.write("This dashboard shows dataset metrics and a quick random-policy simulation. RL agent not yet integrated.")

DATA_PATH = os.path.join('data', 'processed', 'coffee_loans_hybrid.csv')

if not os.path.exists(DATA_PATH):
    st.warning(f"Processed dataset not found: {DATA_PATH}. Run `python generate_dataset.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Loans", f"{len(df):,}")
col2.metric("Mills", f"{df['mill_id'].nunique() if 'mill_id' in df.columns else 'N/A'}")
col3.metric("Default rate", f"{df['default'].mean():.1%}" if 'default' in df.columns else 'N/A')

st.subheader("Distributions")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
if 'loan_amount_usd' in df.columns:
    axes[0].hist(df['loan_amount_usd'].dropna(), bins=30)
    axes[0].set_title('Loan amount (USD)')
else:
    axes[0].text(0.5, 0.5, 'loan_amount_usd missing', ha='center')

if 'coffee_price' in df.columns:
    axes[1].hist(df['coffee_price'].dropna(), bins=30)
    axes[1].set_title('Coffee price')
else:
    axes[1].text(0.5, 0.5, 'coffee_price missing', ha='center')

st.pyplot(fig)

st.subheader("Sample rows")
st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))

st.subheader("Quick random-policy simulation")
if st.button("Run random simulation (one episode)"):
    try:
        from coffee_env import CoffeeLendingEnv

        env = CoffeeLendingEnv(DATA_PATH)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1

        stats = env.get_episode_stats()
        st.success("Simulation complete")
        st.write(stats)
        st.write(f"Total reward: {total_reward:.3f} over {steps} steps")

    except Exception as e:
        st.error(f"Simulation failed: {e}")

st.markdown("**Status**: RL agent not yet integrated. Use `train_rl.py` to add a training pipeline.")
