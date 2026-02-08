"""
Comprehensive Policy Comparison
Compares RL agents against baseline policies with publication-ready visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import argparse

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from coffee_env import CoffeeLendingEnv
from baseline_policies import (
    RandomPolicy, RejectAllPolicy, ApproveAllPolicy, 
    RuleBasedPolicy, LogisticRegressionPolicy, evaluate_policy
)


def load_rl_policy(model_path):
    """Load trained RL policy"""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required")
    
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    return PPO.load(model_path)


class RLPolicyWrapper:
    """Wrapper to make RL model compatible with baseline policy interface"""
    
    def __init__(self, model, name="PPO"):
        self.model = model
        self.name = name
    
    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action.item() if hasattr(action, 'item') else action)


def create_comparison_plots(results, save_dir):
    """Create publication-ready comparison plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    policy_names = list(results.keys())
    
    # Define color scheme
    colors = {
        'Random': '#e74c3c',
        'Reject-All': '#95a5a6',
        'Approve-All-$50k': '#f39c12',
        'Approve-All-$300k': '#e67e22',
        'Rule-Based': '#3498db',
        'Logistic-Regression': '#9b59b6',
        'PPO': '#2ecc71'
    }
    
    # Get colors for policies
    plot_colors = [colors.get(name, '#34495e') for name in policy_names]
    
    # Figure 1: Main Comparison (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Compound Returns
    returns = [results[p]['mean_return'] * 100 for p in policy_names]
    return_stds = [results[p]['std_return'] * 100 for p in policy_names]
    
    bars = axes[0, 0].bar(range(len(policy_names)), returns, yerr=return_stds,
                          color=plot_colors, alpha=0.7, edgecolor='black', capsize=5)
    axes[0, 0].set_title('Portfolio Returns', fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].set_xticks(range(len(policy_names)))
    axes[0, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Default Rates
    defaults = [results[p]['mean_defaults'] * 100 for p in policy_names]
    
    bars = axes[0, 1].bar(range(len(policy_names)), defaults,
                          color=plot_colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Default Rates', fontweight='bold')
    axes[0, 1].set_ylabel('Default Rate (%)')
    axes[0, 1].set_xticks(range(len(policy_names)))
    axes[0, 1].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 1].axhline(15, color='orange', linestyle='--', alpha=0.7, 
                       linewidth=1.5, label='Dataset Avg (15%)')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Risk-Adjusted Returns (Sharpe Ratio)
    sharpe = [results[p]['sharpe_ratio'] for p in policy_names]
    
    bars = axes[1, 0].bar(range(len(policy_names)), sharpe,
                          color=plot_colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_xticks(range(len(policy_names)))
    axes[1, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Social Impact (Farmers Helped)
    farmers = [results[p]['mean_farmers'] / 1000 for p in policy_names]
    
    bars = axes[1, 1].bar(range(len(policy_names)), farmers,
                          color=plot_colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Social Impact', fontweight='bold')
    axes[1, 1].set_ylabel('Farmers Helped (thousands)')
    axes[1, 1].set_xticks(range(len(policy_names)))
    axes[1, 1].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}k', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'policy_comparison_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Detailed Performance Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    returns = [results[p]['mean_return'] * 100 for p in policy_names]
    defaults = [results[p]['mean_defaults'] * 100 for p in policy_names]
    sizes = [results[p]['mean_approved'] for p in policy_names]
    
    scatter = ax.scatter(defaults, returns, s=[s/2 for s in sizes], 
                        c=plot_colors, alpha=0.6, edgecolors='black', linewidths=2)
    
    # Add labels
    for i, name in enumerate(policy_names):
        ax.annotate(name, (defaults[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Default Rate (%)', fontweight='bold')
    ax.set_ylabel('Portfolio Return (%)', fontweight='bold')
    ax.set_title('Policy Performance: Risk vs Return', fontweight='bold', fontsize=14)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(15, color='orange', linestyle='--', alpha=0.5, linewidth=1, 
              label='Dataset Default Rate')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Add annotation for bubble size
    ax.text(0.95, 0.05, 'Bubble size = Loans Approved', 
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'policy_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Return Distribution Box Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect episode returns for box plot
    episode_returns = []
    labels = []
    
    for name in policy_names:
        if 'episodes' in results[name] and 'returns' in results[name]['episodes']:
            returns_pct = [r * 100 for r in results[name]['episodes']['returns']]
            episode_returns.append(returns_pct)
            labels.append(name)
    
    if episode_returns:
        bp = ax.boxplot(episode_returns, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], plot_colors[:len(episode_returns)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Portfolio Return (%)', fontweight='bold')
        ax.set_title('Return Distribution Across Episodes', fontweight='bold', fontsize=14)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / 'return_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Plots saved to: {save_dir}")


def create_summary_table(results, save_dir):
    """Create CSV summary table"""
    save_dir = Path(save_dir)
    
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Policy': name,
            'Mean_Return_%': res['mean_return'] * 100,
            'Std_Return_%': res['std_return'] * 100,
            'Min_Return_%': res['min_return'] * 100,
            'Max_Return_%': res['max_return'] * 100,
            'Per_Loan_Return_%': res['per_loan_return'],
            'Sharpe_Ratio': res['sharpe_ratio'],
            'Default_Rate_%': res['mean_defaults'] * 100,
            'Loans_Approved': res['mean_approved'],
            'Farmers_Helped': res['mean_farmers'],
            'Final_Capital_$': res['mean_capital']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Mean_Return_%', ascending=False)
    
    csv_path = save_dir / 'comparison_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    print(f"✓ Summary table saved to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compare RL agents against baseline policies'
    )
    parser.add_argument('--data', type=str, 
                       default='data/processed/coffee_loans_hybrid.csv',
                       help='Path to dataset')
    parser.add_argument('--rl-model', type=str, default=None,
                       help='Path to trained RL model (optional)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-loans', type=int, default=500,
                       help='Maximum loans per episode')
    parser.add_argument('--output', type=str, default='results/comparison',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*90)
    print("COMPREHENSIVE POLICY COMPARISON")
    print("="*90)
    
    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"comparison_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Load environment
    print(f"\n[1/4] Loading environment from: {args.data}")
    env = CoffeeLendingEnv(
        data_path=args.data,
        initial_capital=1_000_000,
        interest_rate=0.25
    )
    env.df = env.df.iloc[:args.max_loans].reset_index(drop=True)
    print(f"  ✓ Environment loaded ({len(env.df)} loans per episode)")
    
    # Load data for ML training
    df = pd.read_csv(args.data)
    
    # Initialize baseline policies
    print(f"\n[2/4] Initializing policies...")
    policies = [
        RandomPolicy(),
        RejectAllPolicy(),
        ApproveAllPolicy(loan_size=1),
        ApproveAllPolicy(loan_size=3),
        RuleBasedPolicy(),
        LogisticRegressionPolicy()
    ]
    
    # Train ML policies
    for policy in policies:
        if isinstance(policy, LogisticRegressionPolicy):
            accuracy = policy.train(df)
            print(f"  ✓ {policy.name}: {accuracy:.2%} test accuracy")
    
    # Add RL policy if provided
    if args.rl_model:
        print(f"\n  Loading RL model: {args.rl_model}")
        try:
            rl_model = load_rl_policy(args.rl_model)
            policies.append(RLPolicyWrapper(rl_model, "PPO"))
            print(f"  ✓ RL model loaded")
        except Exception as e:
            print(f"  ⚠ Could not load RL model: {e}")
    
    # Evaluate all policies
    print(f"\n[3/4] Evaluating {len(policies)} policies ({args.episodes} episodes each)...")
    results = {}
    
    for i, policy in enumerate(policies, 1):
        print(f"  [{i}/{len(policies)}] {policy.name}...", end=' ')
        results[policy.name] = evaluate_policy(policy, env, args.episodes, verbose=False)
        print(f"✓ Return: {results[policy.name]['mean_return']*100:.1f}%")
    
    # Create visualizations and summary
    print(f"\n[4/4] Generating outputs...")
    create_comparison_plots(results, run_dir)
    summary_df = create_summary_table(results, run_dir)
    
    # Save detailed results
    results_file = run_dir / 'detailed_results.json'
    save_data = {}
    for name, res in results.items():
        save_data[name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in res.items() if k != 'episodes'
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_episodes': args.episodes,
            'episode_length': args.max_loans,
            'results': save_data
        }, f, indent=2)
    
    print(f"  ✓ Detailed results: {results_file}")
    
    # Print summary
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*90)
    print("✓ COMPARISON COMPLETE")
    print("="*90)
    print(f"\nOutput directory: {run_dir}")
    print(f"  - policy_comparison_main.png (4-panel comparison)")
    print(f"  - policy_comparison_scatter.png (risk vs return)")
    print(f"  - return_distribution.png (box plots)")
    print(f"  - comparison_summary.csv (detailed metrics)")
    print(f"  - detailed_results.json (raw data)")


if __name__ == '__main__':
    main()