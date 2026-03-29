import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesAgent:
    """
    Agent 1: Ingests historical price data, computes the covariance matrix, 
    and mathematically derives the Composite Instability Index (I_t) 
    using the dominant eigenvalue of the correlation matrix.
    """
    def __init__(self, db_collection):
        self.collection = db_collection

    def execute(self, universe_id, target_date_str, lookback_days=90):
        """
        Executes the agent's deterministic logic for a specific month.
        """
        print(f"🤖 [Agent 1] Waking up. Analyzing {universe_id} for T={target_date_str}")
        
        target_date = pd.to_datetime(target_date_str)
        # Buffer to ensure we get enough trading days (ignoring weekends/holidays)
        start_date = target_date - timedelta(days=lookback_days + 45) 
        
        # 1. Fetch raw prices from MongoDB
        prices_df = self._fetch_price_matrix(universe_id, start_date, target_date)
        
        if prices_df.empty:
            raise ValueError(f"❌ No price data found for {universe_id} ending on {target_date_str}")

        # 2. Calculate daily log returns
        # Forward-fill missing data (e.g., trading halts), then drop remaining NAs
        prices_df = prices_df.ffill().dropna(axis=1) 
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

        # Enforce strict lookback window (grab the last N trading days)
        returns_df = returns_df.tail(lookback_days)

        # 3. Compute Covariance Matrix (\Sigma_t) - Needed for Agent 3 (CVaR)
        covariance_matrix = returns_df.cov()

        # 4. Compute Composite Instability Index (I_t) - The Contagion Metric
        correlation_matrix = returns_df.corr()
        eigenvalues, _ = np.linalg.eigh(correlation_matrix)
        
        # Largest eigenvalue divided by N (Bounds between 0 and 1)
        max_eigenvalue = np.max(eigenvalues)
        N = len(returns_df.columns)
        I_t = max_eigenvalue / N 

        print(f"   -> Extracted {N} valid assets.")
        print(f"   -> Computed Covariance Matrix (Σ_t).")
        print(f"   -> Systemic Instability Index (I_t) = {I_t:.4f}")

        # Return the "Blackboard" state updates
        return {
            "returns_df": returns_df,
            "covariance_matrix": covariance_matrix,
            "instability_index": I_t
        }

    def _fetch_price_matrix(self, universe_id, start_date, end_date):
        """
        Queries MongoDB and pivots the embedded arrays into a clean N x T DataFrame.
        """
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        cursor = self.collection.find({"universes": universe_id}, {"ticker": 1, "historical_prices": 1})
        
        all_prices = {}
        for doc in cursor:
            ticker = doc.get("ticker")
            history = doc.get("historical_prices", [])
            
            if not history:
                continue
            
            # Filter the history list of dicts directly before converting to DataFrame to save memory
            filtered_history = [
                row for row in history 
                if row.get('Date') and start_str <= row['Date'] <= end_str
            ]
            
            if not filtered_history:
                continue
                
            df = pd.DataFrame(filtered_history)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Keep only the Close price
            all_prices[ticker] = df['Close'].astype(float)
                
        if not all_prices:
            return pd.DataFrame()
            
        # Combine into a single matrix (Rows = Dates, Columns = Tickers)
        price_matrix = pd.DataFrame(all_prices)
        return price_matrix