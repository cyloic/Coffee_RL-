"""
Baseline Policies for Coffee Lending
Implements and evaluates non-RL baseline policies from Colab experiments
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime


class BaselinePolicy:
    """Base class for baseline policies"""
    
    def __init__(self, name):
        self.name = name
    
    def predict(self, state):
        """Predict action given state"""
        raise NotImplementedError
    
    def train(self, data):
        """Train policy if needed"""
        pass


class RandomPolicy(BaselinePolicy):
    """Random baseline - selects actions uniformly"""
    
    def __init__(self):
        super().__init__("Random")
    
    def predict(self, state):
        return np.random.randint(0, 4)


class RejectAllPolicy(BaselinePolicy):
    """Conservative baseline - rejects all loans"""
    
    def __init__(self):
        super().__init__("Reject-All")
    
    def predict(self, state):
        return 0


class ApproveAllPolicy(BaselinePolicy):
    """Aggressive baseline - approves maximum loans"""
    
    def __init__(self, loan_size=3):
        super().__init__(f"Approve-All-${[0, 50, 150, 300][loan_size]}k")
        self.loan_size = loan_size
    
    def predict(self, state):
        # Only approve if we have capital (state[9] is capital ratio)
        if state[9] < 0.1:
            return 0
        return self.loan_size


class RuleBasedPolicy(BaselinePolicy):
    """
    Rule-based policy using domain knowledge
    Risk score based on: experience, utilization, certification, coffee price, debt ratio
    """
    
    def __init__(self):
        super().__init__("Rule-Based")
    
    def predict(self, state):
        # Check capital availability (state[9] is capital/initial_capital)
        if state[9] < 0.1:
            return 0
        
        # Calculate risk score (0 = low risk, 1 = high risk)
        # state indices: [years_op, utilization, certification, farmers, 
        #                 coffee_price, rainfall, capacity, debt_ratio, contracts, capital, default_rate]
        risk_score = (
            (1 - state[0]) * 0.25 +  # Less experience = higher risk
            (1 - state[1]) * 0.20 +  # Lower utilization = higher risk
            (1 - state[2]) * 0.20 +  # No certification = higher risk
            (1 - state[4]) * 0.25 +  # Lower coffee price = higher risk
            state[7] * 0.10          # Higher debt = higher risk
        )
        
        # Map risk to loan size
        if risk_score < 0.30:
            return 3  # $300k - low risk
        elif risk_score < 0.50:
            return 2  # $150k - medium risk
        elif risk_score < 0.65:
            return 1  # $50k - higher risk
        else:
            return 0  # Reject - too risky


class LogisticRegressionPolicy(BaselinePolicy):
    """
    ML-based policy using Logistic Regression to predict defaults
    Maps default probability to loan amount
    """
    
    def __init__(self):
        super().__init__("Logistic-Regression")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.feature_indices = [0, 1, 2, 4, 5, 6, 7, 8]  # Exclude capital and default_rate
        self.trained = False
    
    def train(self, data):
        """Train on historical data"""
        # Features used in Colab
        feature_cols = ['years_operating', 'utilization_rate', 'has_certification', 
                       'coffee_price', 'rainfall_mm', 'capacity_tons', 
                       'debt_to_revenue', 'buyer_contracts']
        
        X = data[feature_cols].values
        y = data['default'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.train_accuracy = self.model.score(X_test, y_test)
        self.trained = True
        
        return self.train_accuracy
    
    def predict(self, state):
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Check capital availability
        if state[9] < 0.1:
            return 0
        
        # Extract features (first 8 state components, excluding capital and default_rate)
        features = state[self.feature_indices].reshape(1, -1)
        
        # Get default probability
        default_prob = self.model.predict_proba(features)[0, 1]
        
        # Map probability to loan size (thresholds from Colab)
        if default_prob < 0.15:
            return 3  # $300k - very low risk
        elif default_prob < 0.22:
            return 2  # $150k - low risk
        elif default_prob < 0.30:
            return 1  # $50k - medium risk
        else:
            return 0  # Reject - high risk


def evaluate_policy(policy, env, n_episodes=20, verbose=False):
    """
    Evaluate a policy over multiple episodes
    
    Returns:
        dict: Aggregated statistics
    """
    results = {
        'returns': [],
        'defaults': [],
        'approved': [],
        'farmers': [],
        'final_capital': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        
        while not done:
            action = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        stats = env.get_episode_stats()
        results['returns'].append(stats['portfolio_return'])
        results['defaults'].append(stats['default_rate'])
        results['approved'].append(stats['total_loans_approved'])
        results['farmers'].append(stats['farmers_helped'])
        results['final_capital'].append(stats['final_capital'])
        
        if verbose and ep < 3:
            print(f"  Episode {ep+1}: Return={stats['portfolio_return']:.2%}, "
                  f"Capital=${stats['final_capital']:,.0f}")
    
    # Calculate metrics
    avg_return = np.mean(results['returns'])
    avg_approved = np.mean(results['approved'])
    
    # Per-loan return
    if avg_approved > 0:
        per_loan_return = (((1 + avg_return) ** (1/avg_approved)) - 1) * 100
    else:
        per_loan_return = 0
    
    # Sharpe ratio (risk-adjusted return)
    if np.std(results['returns']) > 0:
        sharpe = np.mean(results['returns']) / np.std(results['returns'])
    else:
        sharpe = 0
    
    return {
        'mean_return': np.mean(results['returns']),
        'std_return': np.std(results['returns']),
        'min_return': np.min(results['returns']),
        'max_return': np.max(results['returns']),
        'per_loan_return': per_loan_return,
        'sharpe_ratio': sharpe,
        'mean_defaults': np.mean(results['defaults']),
        'mean_approved': np.mean(results['approved']),
        'mean_farmers': np.mean(results['farmers']),
        'mean_capital': np.mean(results['final_capital']),
        'episodes': results
    }


def compare_baselines(env, data_path, n_episodes=20, save_results=True):
    """
    Compare all baseline policies
    
    Args:
        env: Coffee lending environment
        data_path: Path to dataset for training ML models
        n_episodes: Number of evaluation episodes
        save_results: Whether to save results to file
    """
    print("="*90)
    print("BASELINE POLICY COMPARISON")
    print("="*90)
    print(f"Environment: {len(env.df)} loans per episode")
    print(f"Evaluation: {n_episodes} episodes each\n")
    
    # Load data for ML training
    df = pd.read_csv(data_path)
    
    # Initialize policies
    policies = [
        RandomPolicy(),
        RejectAllPolicy(),
        ApproveAllPolicy(loan_size=1),  # $50k
        ApproveAllPolicy(loan_size=3),  # $300k
        RuleBasedPolicy(),
        LogisticRegressionPolicy()
    ]
    
    # Train ML policies
    print("[1/2] Training ML policies...")
    for policy in policies:
        if isinstance(policy, LogisticRegressionPolicy):
            accuracy = policy.train(df)
            print(f"  {policy.name}: {accuracy:.2%} accuracy on test set")
    
    print("\n[2/2] Evaluating policies...")
    results = {}
    
    for policy in policies:
        print(f"\n  Testing {policy.name}...")
        results[policy.name] = evaluate_policy(policy, env, n_episodes, verbose=False)
    
    # Print comparison table
    print("\n" + "="*90)
    print("RESULTS: COMPOUND RETURNS (over full episode)")
    print("="*90)
    print(f"{'Policy':<20} {'Return':<15} {'Std Dev':<12} {'Default Rate':<14} {'Loans Approved'}")
    print("-"*90)
    
    for name, res in results.items():
        print(f"{name:<20} {res['mean_return']*100:>6.1f}% {'':<7} "
              f"±{res['std_return']*100:>5.1f}% {'':<3} "
              f"{res['mean_defaults']*100:>6.1f}% {'':<6} "
              f"{res['mean_approved']:>7.0f}")
    
    # Per-loan returns
    print("\n" + "="*90)
    print("RESULTS: PER-LOAN RETURNS (normalized)")
    print("="*90)
    print(f"{'Policy':<20} {'Per-Loan Return':<20} {'Farmers Helped'}")
    print("-"*90)
    
    for name, res in results.items():
        print(f"{name:<20} {res['per_loan_return']:>6.3f}% {'':<12} "
              f"{res['mean_farmers']:>10,.0f}")
    
    # Risk-adjusted
    print("\n" + "="*90)
    print("RESULTS: RISK-ADJUSTED PERFORMANCE")
    print("="*90)
    print(f"{'Policy':<20} {'Sharpe Ratio':<15} {'Min Return':<15} {'Max Return'}")
    print("-"*90)
    
    for name, res in results.items():
        print(f"{name:<20} {res['sharpe_ratio']:>6.2f} {'':<8} "
              f"{res['min_return']*100:>6.1f}% {'':<7} "
              f"{res['max_return']*100:>6.1f}%")
    
    print("\n" + "="*90)
    
    # Save results
    if save_results:
        output_dir = Path('results') / 'baselines'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'baseline_comparison_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
        save_data = {}
        for name, res in results.items():
            save_data[name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in res.items() if k != 'episodes'
            }
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'n_episodes': n_episodes,
                'results': save_data
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    from coffee_env import CoffeeLendingEnv
    
    parser = argparse.ArgumentParser(description='Evaluate baseline policies')
    parser.add_argument('--data', type=str, 
                       default='data/processed/coffee_loans_hybrid.csv',
                       help='Path to dataset')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-loans', type=int, default=500,
                       help='Maximum loans per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    
    # Create environment
    print(f"Loading environment from: {args.data}")
    env = CoffeeLendingEnv(
        data_path=args.data,
        initial_capital=1_000_000,
        interest_rate=0.25
    )
    
    # Limit episode length
    env.df = env.df.iloc[:args.max_loans].reset_index(drop=True)
    print(f"Episode length: {len(env.df)} loans\n")
    
    # Run comparison
    results = compare_baselines(env, args.data, n_episodes=args.episodes)
    
    print("\n" + "="*90)
    print("✓ BASELINE EVALUATION COMPLETE")
    print("="*90)