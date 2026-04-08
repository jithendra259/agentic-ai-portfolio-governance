import time

import cvxpy as cp
import numpy as np
import pandas as pd


class GCVaROptimizerAgent:
    """
    Agent 3: Notebook-aligned G-CVaR optimizer with benchmark portfolios and
    governance trigger metadata.
    """

    def __init__(
        self,
        alpha=0.95,
        lambda_max=1.0,
        k=10,
        I_thresh=0.85,
        tau_crisis=0.85,
        tau_turnover=0.40,
        max_weight=0.15,
    ):
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.k = k
        self.I_thresh = I_thresh
        self.tau_crisis = tau_crisis
        self.tau_turnover = tau_turnover
        self.max_weight = max_weight

    def execute(self, returns_df, c_vector, I_t, previous_weights=None):
        print("[Agent 3] Running notebook-aligned G-CVaR optimization...")

        valid_assets = returns_df.columns.intersection(c_vector.index)
        if len(valid_assets) == 0:
            raise ValueError("Alignment error: no overlapping assets between returns and graph.")

        aligned_returns = returns_df[valid_assets].dropna(how="any")
        if aligned_returns.empty:
            raise ValueError("Optimization failed: aligned returns are empty after dropping NaNs.")

        returns_matrix = aligned_returns.to_numpy(dtype=float)
        tickers = valid_assets.tolist()
        num_periods, num_assets = returns_matrix.shape
        graph_scores = c_vector[valid_assets].astype(float).to_numpy()

        lambda_t = float(self.lambda_max / (1.0 + np.exp(-self.k * (I_t - self.I_thresh))))
        effective_max_weight = self._effective_max_weight(num_assets)
        print(
            f"   -> Instability (I_t): {I_t:.4f} | Graph penalty (lambda_t): {lambda_t:.4f} "
            f"| Max weight: {effective_max_weight:.4f}"
        )

        started_at = time.perf_counter()
        gcvar_weights, solver_status, solver_name, cvar_objective = self._solve_gcvar(
            returns_matrix,
            graph_scores,
            lambda_t,
            effective_max_weight,
            tickers,
        )
        std_cvar_weights = self._solve_standard_cvar(returns_matrix, effective_max_weight, tickers)
        mean_variance_weights = self._solve_mean_variance(aligned_returns, effective_max_weight, tickers)
        equal_weight_weights = pd.Series(np.repeat(1.0 / num_assets, num_assets), index=tickers, dtype=float)

        turnover_value = self._compute_turnover(
            current_weights=gcvar_weights,
            previous_weights=previous_weights,
            tickers=tickers,
        )
        hitl_crisis = bool(I_t >= self.tau_crisis)
        hitl_turnover = bool(turnover_value > self.tau_turnover)
        hitl_required = bool(hitl_crisis or hitl_turnover)
        hitl_reasons = []
        if hitl_crisis:
            hitl_reasons.append(f"crisis: I_t={I_t:.4f} >= {self.tau_crisis:.2f}")
        if hitl_turnover:
            hitl_reasons.append(f"turnover: {turnover_value:.4f} > {self.tau_turnover:.2f}")

        solve_time_s = float(time.perf_counter() - started_at)
        print(f"   -> Optimization complete. Solver={solver_name} | HITL={hitl_required}")

        return {
            "optimal_weights": gcvar_weights,
            "strategy_weights": {
                "g_cvar": gcvar_weights,
                "std_cvar": std_cvar_weights,
                "mean_variance": mean_variance_weights,
                "equal_weight": equal_weight_weights,
            },
            "lambda_t": lambda_t,
            "cvar_objective": cvar_objective,
            "turnover": {"g_cvar": turnover_value},
            "hitl_required": hitl_required,
            "hitl_crisis": hitl_crisis,
            "hitl_turnover": hitl_turnover,
            "hitl_reasons": hitl_reasons,
            "solver_status": solver_status,
            "solver_name": solver_name,
            "solve_time_s": solve_time_s,
            "max_weight_constraint": effective_max_weight,
        }

    def _effective_max_weight(self, num_assets):
        minimum_feasible_weight = 1.0 / max(1, num_assets)
        return min(1.0, max(self.max_weight, minimum_feasible_weight))

    def _solve_gcvar(self, returns_matrix, graph_scores, lambda_t, max_weight, tickers):
        num_periods, num_assets = returns_matrix.shape
        weights = cp.Variable(num_assets, nonneg=True)
        zeta = cp.Variable()
        tail_losses = cp.Variable(num_periods, nonneg=True)
        losses = -returns_matrix @ weights

        cvar_term = zeta + (1.0 / (num_periods * (1.0 - self.alpha))) * cp.sum(tail_losses)
        graph_penalty = lambda_t * (graph_scores @ weights)
        problem = cp.Problem(
            cp.Minimize(cvar_term + graph_penalty),
            [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= max_weight,
                tail_losses >= losses - zeta,
                tail_losses >= 0,
            ],
        )

        for solver in [cp.CLARABEL, cp.ECOS, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver)
            except Exception:
                continue

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and weights.value is not None:
                cleaned = self._clean_weights(weights.value, num_assets, tickers)
                return cleaned, str(problem.status), str(solver), float(cvar_term.value)

        fallback = pd.Series(np.repeat(1.0 / num_assets, num_assets), index=tickers, dtype=float)
        return fallback, "fallback_equal_weight", "none", float("nan")

    def _solve_standard_cvar(self, returns_matrix, max_weight, tickers):
        num_periods, num_assets = returns_matrix.shape
        weights = cp.Variable(num_assets, nonneg=True)
        zeta = cp.Variable()
        tail_losses = cp.Variable(num_periods, nonneg=True)
        losses = -returns_matrix @ weights

        problem = cp.Problem(
            cp.Minimize(zeta + (1.0 / (num_periods * (1.0 - self.alpha))) * cp.sum(tail_losses)),
            [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= max_weight,
                tail_losses >= losses - zeta,
                tail_losses >= 0,
            ],
        )

        for solver in [cp.CLARABEL, cp.ECOS, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver)
            except Exception:
                continue

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and weights.value is not None:
                return self._clean_weights(weights.value, num_assets, tickers)

        return pd.Series(np.repeat(1.0 / num_assets, num_assets), index=tickers, dtype=float)

    def _solve_mean_variance(self, returns_df, max_weight, tickers):
        num_assets = returns_df.shape[1]
        covariance = returns_df.cov().to_numpy(dtype=float) + np.eye(num_assets) * 1e-6
        weights = cp.Variable(num_assets, nonneg=True)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(weights, covariance)),
            [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= max_weight,
            ],
        )

        for solver in [cp.CLARABEL, cp.ECOS, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver)
            except Exception:
                continue

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and weights.value is not None:
                return self._clean_weights(weights.value, num_assets, tickers)

        return pd.Series(np.repeat(1.0 / num_assets, num_assets), index=tickers, dtype=float)

    def _clean_weights(self, raw_weights, num_assets, tickers):
        cleaned = pd.Series(np.maximum(np.asarray(raw_weights).reshape(-1), 0.0), index=tickers, dtype=float)
        cleaned = cleaned.apply(lambda value: 0.0 if value < 1e-4 else value)

        weight_sum = float(cleaned.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return pd.Series(np.repeat(1.0 / num_assets, num_assets), index=tickers, dtype=float)

        cleaned = cleaned / weight_sum
        return cleaned

    def _compute_turnover(self, current_weights, previous_weights, tickers):
        if previous_weights is None:
            return 0.0

        if isinstance(previous_weights, pd.Series):
            previous = previous_weights.reindex(tickers).fillna(0.0).astype(float)
        elif isinstance(previous_weights, dict):
            previous = pd.Series(previous_weights, dtype=float).reindex(tickers).fillna(0.0)
        else:
            previous = pd.Series(np.asarray(previous_weights).reshape(-1), dtype=float)
            previous = previous.reindex(range(len(tickers)), fill_value=0.0)

        current = current_weights.copy()
        if len(current.index) != len(tickers):
            current.index = tickers

        return float(np.abs(current.values - previous.values).sum() / 2.0)
