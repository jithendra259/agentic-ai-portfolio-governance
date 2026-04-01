import cvxpy as cp
import numpy as np
import pandas as pd


class GCVaROptimizerAgent:
    """
    Agent 3: Minimize CVaR subject to the graph-regularized penalty.
    """

    def __init__(self, alpha=0.95, lambda_max=0.5, k=10, I_thresh=0.5):
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.k = k
        self.I_thresh = I_thresh

    def execute(self, returns_df, c_vector, I_t):
        print("[Agent 3] Running graph-regularized CVaR optimization...")

        valid_assets = returns_df.columns.intersection(c_vector.index)
        if len(valid_assets) == 0:
            raise ValueError("Alignment error: no overlapping assets between returns and graph.")

        returns_matrix = returns_df[valid_assets].values
        graph_scores = c_vector[valid_assets].values
        tickers = valid_assets.tolist()
        num_periods, num_assets = returns_matrix.shape

        lambda_t = self.lambda_max / (1 + np.exp(-self.k * (I_t - self.I_thresh)))
        print(f"   -> Instability (I_t): {I_t:.4f} | Graph penalty (lambda_t): {lambda_t:.4f}")

        weights = cp.Variable(num_assets)
        gamma = cp.Variable()
        tail_losses = cp.Variable(num_periods)

        cvar_term = gamma + (1.0 / (num_periods * (1 - self.alpha))) * cp.sum(tail_losses)
        graph_penalty = lambda_t * (weights @ graph_scores)
        objective = cp.Minimize(cvar_term + graph_penalty)

        constraints = [
            tail_losses >= 0,
            tail_losses >= -returns_matrix @ weights - gamma,
            cp.sum(weights) == 1,
            weights >= 0,
        ]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except Exception as exc:
            print(f"WARNING: ECOS failed, falling back to SCS: {exc}")
            problem.solve(solver=cp.SCS)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
            raise ValueError(
                f"Optimization failed. Solver could not find an optimal weight vector. Status: {problem.status}"
            )

        optimal_weights = pd.Series(
            np.maximum(np.asarray(weights.value).reshape(-1), 0.0),
            index=tickers,
        )
        optimal_weights = optimal_weights.apply(lambda value: 0.0 if value < 1e-4 else value)

        weight_sum = float(optimal_weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("Weight normalization failed after cleanup: no positive weights remained.")

        optimal_weights = optimal_weights / weight_sum
        if not np.isclose(optimal_weights.sum(), 1.0):
            raise ValueError("Weight normalization failed: cleaned weights do not sum to 1.0.")
        if not np.all(optimal_weights.values >= 0.0):
            raise ValueError("Negative weights detected after cleanup.")

        print("   -> Optimization complete.")

        return {
            "optimal_weights": optimal_weights,
            "lambda_t": lambda_t,
            "cvar_objective": cvar_term.value,
        }
