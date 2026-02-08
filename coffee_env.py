"""
Coffee Supply Chain Lending Environment - FIXED VERSION
Handles NaN, bounds, and negative capital properly
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

class CoffeeLendingEnv(gym.Env):
    """RL Environment for Coffee Mill Lending"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data_path: str, initial_capital: float = 1_000_000, interest_rate: float = 0.25):
        super().__init__()
        
        self.df = pd.read_csv(data_path)
        self.initial_capital = initial_capital
        self.interest_rate = interest_rate
        
        self.loan_amounts = {0: 0, 1: 50_000, 2: 150_000, 3: 300_000}
        self.action_space = spaces.Discrete(4)
        
        # FIXED: Observation space allows negative values for capital ratio
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(11,), dtype=np.float32)
        
        self._compute_normalization()
        self.reset()
        
    def _compute_normalization(self):
        """Compute min/max for normalization - with NaN handling"""
        self.norm = {}
        
        for col in ['years_operating', 'num_farmers', 'coffee_price', 'rainfall_mm', 
                    'capacity_tons', 'debt_to_revenue', 'buyer_contracts']:
            if col in self.df.columns:
                # Handle NaN values
                valid_data = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    min_val = float(valid_data.min())
                    max_val = float(valid_data.max())
                else:
                    min_val, max_val = 0.0, 1.0
            else:
                min_val, max_val = 0.0, 1.0
            
            self.norm[col] = (min_val, max_val)
    
    def _normalize(self, value: float, feature: str) -> float:
        """Normalize value with safety checks"""
        # Handle NaN/inf
        if pd.isna(value) or np.isinf(value):
            return 0.5
        
        min_val, max_val = self.norm[feature]
        
        if max_val == min_val:
            return 0.5
        
        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _get_state(self) -> np.ndarray:
        """Get current state with proper bounds and NaN handling"""
        if self.current_step >= len(self.df):
            return np.zeros(11, dtype=np.float32)
        
        loan = self.df.iloc[self.current_step]
        
        # Extract features with safe defaults
        state = [
            self._normalize(loan.get('years_operating', 0), 'years_operating'),
            float(np.clip(loan.get('utilization_rate', 0.5), 0.0, 1.0)),
            float(loan.get('has_certification', 0)),
            self._normalize(loan.get('num_farmers', 0), 'num_farmers'),
            self._normalize(loan.get('coffee_price', 0), 'coffee_price'),
            self._normalize(loan.get('rainfall_mm', 0), 'rainfall_mm'),
            self._normalize(loan.get('capacity_tons', 0), 'capacity_tons'),
            self._normalize(loan.get('debt_to_revenue', 0), 'debt_to_revenue'),
            self._normalize(loan.get('buyer_contracts', 0), 'buyer_contracts'),
        ]
        
        # FIXED: Capital ratio can be negative or > 1
        # Clip to reasonable range [-2, 3] to represent -200% to +300%
        capital_ratio = self.capital / self.initial_capital
        capital_ratio = float(np.clip(capital_ratio, -2.0, 3.0))
        state.append(capital_ratio)
        
        # FIXED: Default rate with safety
        if self.total_approvals > 0:
            default_rate = self.total_defaults / self.total_approvals
        else:
            default_rate = 0.0
        default_rate = float(np.clip(default_rate, 0.0, 1.0))
        state.append(default_rate)
        
        # Convert to array and final safety check
        state_array = np.array(state, dtype=np.float32)
        
        # Replace any remaining NaN/inf with 0
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return state_array
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        self.current_step = 0
        self.capital = self.initial_capital
        self.total_defaults = 0
        self.total_approvals = 0
        self.total_farmers_helped = 0
        
        return self._get_state(), {'capital': self.capital}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Convert action
        if hasattr(action, 'item'):
            action = int(action.item())
        elif isinstance(action, np.ndarray):
            action = int(action.flatten()[0])
        else:
            action = int(action)
        
        # Clip action to valid range
        action = int(np.clip(action, 0, 3))
        
        if self.current_step >= len(self.df):
            return self._get_state(), 0.0, True, False, {}
        
        loan = self.df.iloc[self.current_step]
        loan_amount = self.loan_amounts[action]
        
        reward = 0.0
        info = {'action': action, 'loan_amount': loan_amount, 'defaulted': False, 'capital_before': self.capital}
        
        # REJECT
        if action == 0:
            reward = -10 if loan.get('default', 0) == 0 else 5
        
        # APPROVE
        else:
            if self.capital < loan_amount:
                # Insufficient capital - penalize
                reward = -20
                info['reason'] = 'insufficient_capital'
            else:
                # Deduct capital
                self.capital -= loan_amount
                self.total_approvals += 1
                
                if loan.get('default', 0) == 1:
                    # Default: money is lost (already deducted)
                    reward = -loan_amount / 1000
                    self.total_defaults += 1
                    info['defaulted'] = True
                else:
                    # Repaid: return principal + interest
                    interest = loan_amount * self.interest_rate
                    self.capital += loan_amount + interest
                    reward = interest / 1000
                    
                    farmers = loan.get('num_farmers', 0)
                    self.total_farmers_helped += farmers
                    reward += farmers / 1000
        
        self.current_step += 1
        done = (self.current_step >= len(self.df))
        
        # FIXED: Check for bankruptcy (capital <= 0)
        truncated = (self.capital <= 0)
        if truncated:
            done = True
            reward -= 1000  # Heavy penalty for bankruptcy
        
        info['capital_after'] = self.capital
        info['step'] = self.current_step
        info['default_rate'] = self.total_defaults / max(self.total_approvals, 1) if self.total_approvals > 0 else 0
        info['approvals'] = self.total_approvals
        
        # Final reward clipping for stability
        reward = float(np.clip(reward, -1000, 1000))
        
        return self._get_state(), reward, done, truncated, info
    
    def get_episode_stats(self) -> Dict:
        return {
            'final_capital': self.capital,
            'portfolio_return': (self.capital - self.initial_capital) / self.initial_capital,
            'total_loans_approved': self.total_approvals,
            'total_defaults': self.total_defaults,
            'default_rate': self.total_defaults / max(self.total_approvals, 1) if self.total_approvals > 0 else 0,
            'farmers_helped': self.total_farmers_helped,
            'loans_processed': self.current_step,
            'approval_rate': self.total_approvals / max(self.current_step, 1)
        }


# Test the environment
if __name__ == "__main__":
    import sys
    
    try:
        env = CoffeeLendingEnv(
            data_path='data/processed/coffee_loans_fixed.csv',
            initial_capital=1_000_000,
            interest_rate=0.25
        )
        
        print("✅ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"\n✅ Reset successful")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation: {obs}")
        print(f"Contains NaN: {np.any(np.isnan(obs))}")
        print(f"Contains Inf: {np.any(np.isinf(obs))}")
        print(f"Min value: {obs.min():.4f}")
        print(f"Max value: {obs.max():.4f}")
        
        # Test 10 steps
        print("\n✅ Testing 10 steps:")
        for i in range(10):
            action = i % 4
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Capital=${info['capital_after']:,.0f}, NaN={np.any(np.isnan(obs))}")
            
            if done or truncated:
                print("Episode ended early")
                break
        
        print("\n✅ Environment test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)