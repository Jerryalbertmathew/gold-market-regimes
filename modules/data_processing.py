import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def fetch_and_prepare_data(start_date="2010-01-01", end_date="2026-03-01"):
    """
    Downloads historical data for Gold and S&P 500, engineers stationary 
    financial indicators, and applies Robust Scaling for unsupervised learning.
    """
    print("Step 1: Downloading historical data via yfinance...")
    tickers = ["GC=F", "^GSPC"]
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    # Extract closing prices and drop rows with missing data
    prices = raw_data['Close'].dropna()
    
    print("Step 2: Engineering financial features...")
    features = pd.DataFrame(index=prices.index)
    
    # 1. Daily Log Returns (Stationary price tracking)
    features['gold_returns'] = np.log(prices['GC=F'] / prices['GC=F'].shift(1))
    features['sp500_returns'] = np.log(prices['^GSPC'] / prices['^GSPC'].shift(1))
    
    # 2. 20-day Rolling Volatility (Tracks market anxiety)
    features['gold_volatility'] = features['gold_returns'].rolling(window=20).std()
    features['sp500_volatility'] = features['sp500_returns'].rolling(window=20).std()
    
    # 3. Gold-to-Equity Ratio (Tracks macro asset allocation)
    features['gold_sp500_ratio'] = prices['GC=F'] / prices['^GSPC']
    
    # Drop rows containing NaNs created by rolling windows and shifts
    clean_features = features.dropna()
    clean_prices = prices.loc[clean_features.index]
    
    print("Step 3: Standardizing features using RobustScaler...")
    # RobustScaler handles financial outliers much better than standard scaling
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(clean_features)
    
    # Convert scaled array back into a readable DataFrame
    scaled_features = pd.DataFrame(
        scaled_array, 
        index=clean_features.index, 
        columns=clean_features.columns
    )
    
    print("--- Data Preparation Pipeline Complete ---")
    return clean_prices, scaled_features

if __name__ == "__main__":
    prices_df, scaled_df = fetch_and_prepare_data()
    print(scaled_df.tail())
    
