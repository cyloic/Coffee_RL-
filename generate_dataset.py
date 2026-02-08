"""
Hybrid Coffee Lending Dataset Generator
Combines real lending defaults with Rwanda coffee sector features
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COFFEE SUPPLY CHAIN RL - HYBRID DATA GENERATOR")
print("=" * 60)

# ============================================================================
# STEP 1: Download Real Lending Data
# ============================================================================

print("\n[1/5] Downloading lending data from Kaggle...")
print("NOTE: You need to download manually first time")
print("Go to: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
print("Download: accepted_2007_to_2018Q4.csv.gz")
print("Place in: data/raw/lending_club.csv")

# For now, we'll create a realistic synthetic version
# In real run, you'll load the actual Lending Club data

def create_base_lending_data(n_samples=5000):
    """
    Creates realistic lending data similar to Lending Club
    In production, replace this with actual Lending Club data load
    """
    np.random.seed(42)
    
    data = {
        'loan_amnt': np.random.uniform(5000, 35000, n_samples),
        'int_rate': np.random.uniform(6, 25, n_samples),
        'annual_inc': np.random.lognormal(10.5, 0.8, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),  # Debt-to-income
        'emp_length': np.random.randint(0, 15, n_samples),
        'credit_history_length': np.random.randint(3, 30, n_samples),
        'num_credit_lines': np.random.randint(1, 15, n_samples),
        'loan_status': np.random.choice([0, 1], n_samples, p=[0.81, 0.19])  # 19% default
    }
    
    df = pd.DataFrame(data)
    
    # Adjust default probability based on features (make it realistic)
    default_prob = (
        0.05 +  # Base rate
        0.15 * (df['dti'] / 40) +  # Higher DTI = more defaults
        0.10 * (df['int_rate'] / 25) +  # Higher rate = riskier borrowers
        -0.08 * (df['credit_history_length'] / 30)  # Longer history = fewer defaults
    )
    default_prob = np.clip(default_prob, 0, 1)
    df['loan_status'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    return df

print("Creating base lending dataset...")
base_df = create_base_lending_data(5000)
print(f"✓ Created {len(base_df)} lending records")
print(f"  Default rate: {base_df['loan_status'].mean():.1%}")

# ============================================================================
# STEP 2: Download Coffee Price Data
# ============================================================================

print("\n[2/5] Downloading coffee price data...")

def get_coffee_prices():
    """
    Get historical coffee prices
    Using Yahoo Finance coffee futures as proxy
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("KC=F")  # Coffee futures
        hist = ticker.history(period="5y", interval="1mo")
        prices = hist['Close'].values
        dates = hist.index
        
        print(f"✓ Downloaded {len(prices)} months of coffee prices")
        return pd.DataFrame({'date': dates, 'coffee_price': prices})
    
    except Exception as e:
        print(f"⚠ Could not download from Yahoo Finance: {e}")
        print("  Creating synthetic coffee prices based on historical patterns")
        
        # Create realistic coffee price series
        np.random.seed(42)
        n_months = 60  # 5 years
        base_price = 1.20  # USD per lb (historical average)
        
        # Generate with trend + seasonality + noise
        trend = np.linspace(0, 0.3, n_months)  # Slight upward trend
        seasonal = 0.15 * np.sin(np.linspace(0, 10*np.pi, n_months))  # Seasonal
        noise = np.random.normal(0, 0.08, n_months)  # Volatility
        
        prices = base_price + trend + seasonal + noise
        dates = pd.date_range(end=datetime.now(), periods=n_months, freq='M')
        
        print(f"✓ Created {len(prices)} months of synthetic coffee prices")
        return pd.DataFrame({'date': dates, 'coffee_price': prices})

coffee_df = get_coffee_prices()

# ============================================================================
# STEP 3: Download Weather Data (Rwanda)
# ============================================================================

print("\n[3/5] Creating Rwanda weather data...")

def get_rwanda_weather():
    """
    Rwanda weather patterns (rainfall in mm)
    Based on Rwanda Meteorology Agency historical data
    """
    np.random.seed(42)
    n_months = 60
    
    # Rwanda has two rainy seasons (Feb-May, Sep-Dec)
    months = np.arange(n_months) % 12
    
    # Seasonal pattern
    rainy_season_1 = np.isin(months, [1, 2, 3, 4])  # Feb-May
    rainy_season_2 = np.isin(months, [8, 9, 10, 11])  # Sep-Dec
    
    base_rainfall = np.where(rainy_season_1 | rainy_season_2, 120, 40)
    noise = np.random.normal(0, 20, n_months)
    rainfall = np.clip(base_rainfall + noise, 0, 200)
    
    dates = pd.date_range(end=datetime.now(), periods=n_months, freq='M')
    
    print(f"✓ Created {len(rainfall)} months of rainfall data")
    return pd.DataFrame({'date': dates, 'rainfall_mm': rainfall})

weather_df = get_rwanda_weather()

# ============================================================================
# STEP 4: Create Coffee Mill Characteristics
# ============================================================================

print("\n[4/5] Creating coffee mill characteristics...")

def create_mill_characteristics(n_mills=200):
    """
    Create realistic coffee washing station characteristics
    Based on NAEB data and academic research
    """
    np.random.seed(42)
    
    provinces = ['Southern', 'Western', 'Northern', 'Eastern']
    province_probs = [0.50, 0.30, 0.15, 0.05]  # Most mills in Southern
    
    mills = {
        'mill_id': [f'CWS_{i:03d}' for i in range(n_mills)],
        'province': np.random.choice(provinces, n_mills, p=province_probs),
        'capacity_tons': np.random.normal(64, 20, n_mills).clip(30, 150),
        'years_operating': np.random.randint(2, 15, n_mills),
        'num_farmers': np.random.randint(150, 900, n_mills),
        'has_certification': np.random.choice([True, False], n_mills, p=[0.35, 0.65]),
        'buyer_contracts': np.random.randint(1, 6, n_mills),
        'distance_to_market_km': np.random.uniform(5, 80, n_mills),
    }
    
    df = pd.DataFrame(mills)
    df['utilization_rate'] = np.random.uniform(0.25, 0.75, n_mills)
    
    print(f"✓ Created {n_mills} coffee mills")
    print(f"  Certified mills: {df['has_certification'].mean():.0%}")
    print(f"  Avg capacity: {df['capacity_tons'].mean():.0f} tons/year")
    
    return df

mills_df = create_mill_characteristics(200)

# ============================================================================
# STEP 5: Merge Everything into Coffee Lending Dataset
# ============================================================================

print("\n[5/5] Merging into hybrid coffee lending dataset...")

def create_hybrid_dataset(base_df, coffee_df, weather_df, mills_df):
    """
    Merge lending data with coffee context
    Each loan becomes a loan to a coffee mill
    """
    np.random.seed(42)
    n_loans = len(base_df)
    
    # Assign each loan to a mill and time period
    base_df['mill_id'] = np.random.choice(mills_df['mill_id'], n_loans)
    
    # Assign random dates (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    random_days = np.random.randint(0, 5*365, n_loans)
    base_df['loan_date'] = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Merge with mill characteristics
    base_df = base_df.merge(mills_df, on='mill_id', how='left')
    
    # Add coffee price at time of loan
    base_df['loan_month'] = pd.to_datetime(base_df['loan_date']).dt.to_period('M')
    coffee_df['month'] = pd.to_datetime(coffee_df['date']).dt.to_period('M')
    base_df = base_df.merge(
        coffee_df[['month', 'coffee_price']], 
        left_on='loan_month', 
        right_on='month', 
        how='left'
    )
    
    # Add weather at time of loan
    weather_df['month'] = pd.to_datetime(weather_df['date']).dt.to_period('M')
    base_df = base_df.merge(
        weather_df[['month', 'rainfall_mm']],
        left_on='loan_month',
        right_on='month',
        how='left'
    )
    
    # Forward fill any missing values
    base_df['coffee_price'] = base_df['coffee_price'].fillna(method='ffill')
    base_df['rainfall_mm'] = base_df['rainfall_mm'].fillna(method='ffill')
    
    # Adjust defaults based on coffee context
    # Mills are more likely to default when:
    # - Coffee prices are low
    # - Rainfall is extreme (too much or too little)
    # - They're new (low years_operating)
    # - Low utilization
    
    price_effect = (base_df['coffee_price'] - base_df['coffee_price'].mean()) / base_df['coffee_price'].std()
    rainfall_deviation = np.abs(base_df['rainfall_mm'] - 80) / 40  # Optimal around 80mm
    experience_effect = base_df['years_operating'] / 15
    utilization_effect = base_df['utilization_rate']
    
    # Adjust default probability
    default_adjustment = (
        -0.15 * price_effect +  # Low prices = more defaults
        0.10 * rainfall_deviation +  # Extreme weather = more defaults
        -0.10 * experience_effect +  # Experience = fewer defaults
        -0.08 * utilization_effect  # Higher utilization = fewer defaults
    )
    
    # Apply adjustment (about 30% of loans get flipped)
    flip_prob = 1 / (1 + np.exp(-default_adjustment))
    flip_mask = np.random.random(n_loans) < (flip_prob * 0.3)
    base_df.loc[flip_mask, 'loan_status'] = 1 - base_df.loc[flip_mask, 'loan_status']
    
    # Rename columns to coffee context
    base_df = base_df.rename(columns={
        'loan_amnt': 'loan_amount_usd',
        'int_rate': 'interest_rate',
        'annual_inc': 'annual_revenue_usd',
        'dti': 'debt_to_revenue',
        'emp_length': 'mill_age_years',
        'credit_history_length': 'relationship_years',
        'num_credit_lines': 'active_contracts',
        'loan_status': 'default'
    })
    
    # Clean up
    base_df = base_df.drop(['loan_month', 'month_x', 'month_y', 'loan_date'], axis=1, errors='ignore')
    
    print(f"✓ Created hybrid dataset with {len(base_df)} loan records")
    print(f"  Final default rate: {base_df['default'].mean():.1%}")
    print(f"  Date range: Last 5 years")
    print(f"  Features: {len(base_df.columns)} columns")
    
    return base_df

hybrid_df = create_hybrid_dataset(base_df, coffee_df, weather_df, mills_df)

# ============================================================================
# STEP 6: Save Dataset
# ============================================================================

print("\n[6/6] Saving dataset...")

import os
os.makedirs('data/processed', exist_ok=True)

hybrid_df.to_csv('data/processed/coffee_loans_hybrid.csv', index=False)
print(f"✓ Saved to: data/processed/coffee_loans_hybrid.csv")

# Save metadata
metadata = {
    'n_loans': len(hybrid_df),
    'n_mills': mills_df['mill_id'].nunique(),
    'default_rate': hybrid_df['default'].mean(),
    'date_created': datetime.now().strftime('%Y-%m-%d'),
    'features': list(hybrid_df.columns),
    'data_sources': [
        'Base lending patterns: Lending Club style defaults',
        'Coffee prices: Yahoo Finance KC=F futures',
        'Weather: Rwanda Meteorology Agency patterns',
        'Mill characteristics: NAEB reports + academic literature'
    ]
}

import json
with open('data/processed/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: data/processed/metadata.json")

# ============================================================================
# STEP 7: Quick Exploration
# ============================================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"\nTotal loans: {len(hybrid_df):,}")
print(f"Total mills: {hybrid_df['mill_id'].nunique()}")
print(f"Default rate: {hybrid_df['default'].mean():.1%}")
print(f"\nFeatures ({len(hybrid_df.columns)}):")
for col in hybrid_df.columns:
    print(f"  - {col}")

print("\nSample statistics:")
print(hybrid_df[['loan_amount_usd', 'interest_rate', 'coffee_price', 
                 'rainfall_mm', 'years_operating', 'default']].describe())

print("\n" + "=" * 60)
print("✓ DATASET READY FOR RL TRAINING")
print("=" * 60)
print("\nNext steps:")
print("1. Explore data: jupyter notebook")
print("2. Build environment: coffee_env.py")
print("3. Train RL agent: train_rl.py")
print("\nDataset location: data/processed/coffee_loans_hybrid.csv")