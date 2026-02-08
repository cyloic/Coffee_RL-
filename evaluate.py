"""
Coffee RL Evaluation Script
Loads trained models and generates comprehensive evaluation reports
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not found.")

from coffee_env import CoffeeLendingEnv


def load_model(model_path):
    """Load a trained model"""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3[extra]")
    
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return PPO.load(model_path)


def evaluate_policy(env, model, n_episodes=20, deterministic=True, verbose=True):
    """
    Evaluate a policy over multiple episodes
    
    Returns:
        results: dict with aggregated metrics
        episodes: list of per-episode statistics
    """
    episodes = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0
        step_count = 0
        
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(action.item() if hasattr(action, 'item') else action)
            action_counts[action_int] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        stats = env.get_episode_stats()
        stats['episode'] = ep
        stats['episode_reward'] = episode_reward
        stats['steps'] = step_count
        stats['action_distribution'] = action_counts
        episodes.append(stats)
        
        if verbose:
            print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, "
                  f"Capital=${stats['final_capital']:,.0f}, "
                  f"Default Rate={stats['default_rate']:.2%}")
    
    # Aggregate results
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean([e['episode_reward'] for e in episodes]),
        'std_reward': np.std([e['episode_reward'] for e in episodes]),
        'mean_capital': np.mean([e['final_capital'] for e in episodes]),
        'std_capital': np.std([e['final_capital'] for e in episodes]),
        'mean_portfolio_return': np.mean([e['portfolio_return'] for e in episodes]),
        'mean_loans_approved': np.mean([e['total_loans_approved'] for e in episodes]),
        'mean_default_rate': np.mean([e['default_rate'] for e in episodes]),
        'mean_farmers_helped': np.mean([e['farmers_helped'] for e in episodes]),
        'mean_approval_rate': np.mean([e['approval_rate'] for e in episodes]),
        'episodes': episodes
    }
    
    return results


def random_baseline(env, n_episodes=20):
    """Evaluate random policy baseline"""
    episodes = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        stats = env.get_episode_stats()
        stats['episode_reward'] = episode_reward
        episodes.append(stats)
    
    return {
        'mean_reward': np.mean([e['episode_reward'] for e in episodes]),
        'mean_capital': np.mean([e['final_capital'] for e in episodes]),
        'mean_default_rate': np.mean([e['default_rate'] for e in episodes]),
        'mean_approval_rate': np.mean([e['approval_rate'] for e in episodes]),
    }


def plot_evaluation_results(trained_results, baseline_results, save_dir):
    """Create evaluation plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('mean_reward', 'Mean Episode Reward', axes[0, 0]),
        ('mean_capital', 'Mean Final Capital ($)', axes[0, 1]),
        ('mean_default_rate', 'Mean Default Rate', axes[1, 0]),
        ('mean_approval_rate', 'Mean Approval Rate', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        trained_val = trained_results.get(metric, 0)
        baseline_val = baseline_results.get(metric, 0)
        
        x = ['Trained Policy', 'Random Baseline']
        y = [trained_val, baseline_val]
        
        bars = ax.bar(x, y, color=['#2ecc71', '#e74c3c'])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if 'rate' in metric.lower():
                label = f'{height:.2%}'
            elif 'capital' in metric.lower():
                label = f'${height:,.0f}'
            else:
                label = f'{height:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Episode progression
    if 'episodes' in trained_results:
        episodes = trained_results['episodes']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot([e['episode_reward'] for e in episodes], marker='o')
        axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        axes[0, 1].plot([e['final_capital'] for e in episodes], marker='o', color='green')
        axes[0, 1].axhline(y=1_000_000, color='r', linestyle='--', label='Initial Capital')
        axes[0, 1].set_title('Final Capital per Episode', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Capital ($)')
        axes[0, 1].legend()
        
        axes[1, 0].plot([e['default_rate'] for e in episodes], marker='o', color='red')
        axes[1, 0].set_title('Default Rate per Episode', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Default Rate')
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].plot([e['approval_rate'] for e in episodes], marker='o', color='blue')
        axes[1, 1].set_title('Approval Rate per Episode', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Approval Rate')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'episode_progression.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Plots saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Coffee RL model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip)')
    parser.add_argument('--data', type=str, default='data/processed/coffee_loans_hybrid.csv',
                        help='Path to dataset')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--baseline', action='store_true',
                        help='Also evaluate random baseline')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COFFEE RL MODEL EVALUATION")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = output_dir / f"eval_{timestamp}"
    eval_dir.mkdir(exist_ok=True)
    
    # Load environment
    print(f"\n[1/4] Loading environment from: {args.data}")
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        return
    
    env = CoffeeLendingEnv(data_path=args.data)
    print(f"  ✓ Environment loaded ({len(env.df)} loans)")
    
    # Load model
    print(f"\n[2/4] Loading trained model: {args.model}")
    try:
        model = load_model(args.model)
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Evaluate trained policy
    print(f"\n[3/4] Evaluating trained policy ({args.episodes} episodes)...")
    trained_results = evaluate_policy(env, model, n_episodes=args.episodes, verbose=True)
    
    print(f"\n  Trained Policy Results:")
    print(f"  - Mean Reward: {trained_results['mean_reward']:.2f} ± {trained_results['std_reward']:.2f}")
    print(f"  - Mean Capital: ${trained_results['mean_capital']:,.0f} ± ${trained_results['std_capital']:,.0f}")
    print(f"  - Portfolio Return: {trained_results['mean_portfolio_return']:.2%}")
    print(f"  - Default Rate: {trained_results['mean_default_rate']:.2%}")
    print(f"  - Approval Rate: {trained_results['mean_approval_rate']:.2%}")
    print(f"  - Farmers Helped: {trained_results['mean_farmers_helped']:.0f}")
    
    # Evaluate baseline if requested
    baseline_results = None
    if args.baseline:
        print(f"\n[4/4] Evaluating random baseline ({args.episodes} episodes)...")
        baseline_results = random_baseline(env, n_episodes=args.episodes)
        
        print(f"\n  Random Baseline Results:")
        print(f"  - Mean Reward: {baseline_results['mean_reward']:.2f}")
        print(f"  - Mean Capital: ${baseline_results['mean_capital']:,.0f}")
        print(f"  - Default Rate: {baseline_results['mean_default_rate']:.2%}")
        print(f"  - Approval Rate: {baseline_results['mean_approval_rate']:.2%}")
        
        # Calculate improvement
        reward_improvement = (trained_results['mean_reward'] - baseline_results['mean_reward']) / abs(baseline_results['mean_reward']) * 100
        capital_improvement = (trained_results['mean_capital'] - baseline_results['mean_capital']) / baseline_results['mean_capital'] * 100
        
        print(f"\n  Improvement over Random:")
        print(f"  - Reward: {reward_improvement:+.1f}%")
        print(f"  - Capital: {capital_improvement:+.1f}%")
    
    # Save results
    results_file = eval_dir / 'evaluation_results.json'
    save_data = {
        'model_path': args.model,
        'timestamp': timestamp,
        'n_episodes': args.episodes,
        'trained_policy': {k: v for k, v in trained_results.items() if k != 'episodes'},
        'episodes': trained_results.get('episodes', [])
    }
    
    if baseline_results:
        save_data['random_baseline'] = baseline_results
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n  ✓ Results saved to: {results_file}")
    
    # Generate plots
    if baseline_results:
        print(f"\n  Generating comparison plots...")
        plot_evaluation_results(trained_results, baseline_results, eval_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {eval_dir}")
    print(f"  - evaluation_results.json (detailed metrics)")
    if baseline_results:
        print(f"  - policy_comparison.png (trained vs baseline)")
        print(f"  - episode_progression.png (episode-by-episode)")


if __name__ == '__main__':
    main()