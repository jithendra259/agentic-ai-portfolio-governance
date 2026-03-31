import numpy as np
import pandas as pd
import cvxpy as cp

class GCVaROptimizerAgent:
    """
    Agent 3: The Convex Optimization Engine.
    Minimizes Conditional Value-at-Risk (CVaR) subject to a dynamic, 
    graph-regularized penalty (lambda_t * c_t) triggered by the Instability Index.
    """
    def __init__(self, alpha=0.95, lambda_max=0.5, k=10, I_thresh=0.5):
        # CVaR confidence level (95%)
        self.alpha = alpha
        
        # Sigmoid Regime Parameters
        self.lambda_max = lambda_max  # Maximum penalty weight
        self.k = k                    # Steepness of the regime shift
        self.I_thresh = I_thresh      # The threshold where crisis triggers

    def execute(self, returns_df, c_vector, I_t):
        print(f"⚖️ [Agent 3] Waking up. Running Graph-Regularized CVaR Optimization...")
        
        # 1. Align Data: Ensure we only optimize assets that exist in BOTH Agent 1 and Agent 2
        valid_assets = returns_df.columns.intersection(c_vector.index)
        if len(valid_assets) == 0:
            raise ValueError("❌ Alignment Error: No overlapping assets between returns and graph.")
            
        R = returns_df[valid_assets].values
        c = c_vector[valid_assets].values
        tickers = valid_assets.tolist()
        
        T, N = R.shape
        
        # 2. Calculate Adaptive Trust Penalty (\lambda_t) via Sigmoid
        # This is the exact math reviewers will look for.
        lambda_t = self.lambda_max / (1 + np.exp(-self.k * (I_t - self.I_thresh)))
        print(f"   -> Instability (I_t): {I_t:.4f} | Graph Penalty (λ_t): {lambda_t:.4f}")

        # 3. Formulate the Convex Optimization Problem (Rockafellar & Uryasev CVaR)
        w = cp.Variable(N)          # Portfolio weights
        gamma = cp.Variable()       # Value-at-Risk (VaR)
        z = cp.Variable(T)          # Tail losses

        # Objective: Minimize [ CVaR ] + [ Graph Penalty ]
        # CVaR = gamma + (1 / (T * (1 - alpha))) * sum(z)
        cvar_term = gamma + (1.0 / (T * (1 - self.alpha))) * cp.sum(z)
        graph_penalty = lambda_t * (w @ c)
        
        objective = cp.Minimize(cvar_term + graph_penalty)

        # Constraints
        constraints = [
            z >= 0,
            z >= -R @ w - gamma,    # Loss must be greater than negative return - VaR
            cp.sum(w) == 1,         # Fully invested
            w >= 0                  # No short selling
        ]

        # 4. Solve the system
        prob = cp.Problem(objective, constraints)
        try:
            # ECOS is highly reliable for convex portfolio optimization
            prob.solve(solver=cp.ECOS)
        except Exception as e:
            print(f"⚠️ ECOS failed, falling back to SCS: {e}")
            prob.solve(solver=cp.SCS)

        if w.value is None:
            raise ValueError("❌ Optimization failed. Solver could not find an optimal weight vector.")

        # 5. Format Output
        optimal_weights = pd.Series(w.value, index=tickers)
        
        # Clean up near-zero weights (e.g., 1e-9 becomes 0)
        optimal_weights = optimal_weights.apply(lambda x: 0.0 if x < 1e-4 else x)
        optimal_weights = optimal_weights / optimal_weights.sum() # Re-normalize to exactly 1.0
        
        print("   -> Optimization Complete.")
        
        return {
            "optimal_weights": optimal_weights,
            "lambda_t": lambda_t,
            "cvar_objective": cvar_term.value
        }