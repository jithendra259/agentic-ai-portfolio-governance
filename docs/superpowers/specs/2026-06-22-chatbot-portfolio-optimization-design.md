# Chatbot Portfolio Optimization Improvement Design

## Scope

Improve only the production chatbot governance optimizer and its responses. Do not modify the exported notebook, the `.ipynb`, report generation, chart behavior, or unrelated optimizer research code.

The active implementation target is the governance path in `backend/src/agents/live_data_tools.py`, with response assembly in `backend/src/orchestrator/chatbot_orchestrator.py`.

## Goals

- Preserve the requested maximum-weight constraint in every successful solution.
- Make solver fallbacks and relaxed return targets explicit and auditable.
- Convert annual return targets to daily targets geometrically.
- Report portfolio concentration and constraint diagnostics alongside weights.
- Report turnover only when previous portfolio weights are available.
- Preserve existing governance plots, session continuity, stock snapshots, and other chart routes.

## Non-Goals

- Replacing the active optimizer with the notebook's quadratic PSD Graph-CVaR protocol.
- Modifying notebook calculations or research results.
- Adding transaction-cost optimization, short selling, leverage, or trading execution.
- Claiming that optimized weights guarantee future performance.

## Optimization Behavior

The chatbot will continue using its current historical CVaR objective plus the adaptive linear graph-centrality penalty. The first solve uses the risk-profile return floor and the effective maximum-weight cap.

If the first solve is infeasible, the optimizer may relax only the return floor. It must retain:

- full investment: `sum(weights) == 1`;
- long-only weights: `weights >= 0`;
- the effective maximum-weight cap;
- CVaR tail-loss constraints.

The optimizer must not remove the maximum-weight constraint. If no capped solution exists after relaxing the return floor, it returns an error instead of silently returning an uncapped portfolio.

Annual return floors use:

```text
daily_target = (1 + annual_target) ** (1 / 252) - 1
```

This conversion is valid only when `annual_target > -1`. Invalid targets produce a clear optimization error.

## Previous Weights and Turnover

The optimizer accepts optional previous portfolio weights. When supplied, they are aligned to the optimized ticker set and normalized if their positive total is nonzero.

Reported one-way turnover is:

```text
turnover = 0.5 * sum(abs(new_weight - previous_weight))
```

This iteration reports turnover but does not add a turnover penalty to the objective. When previous weights are absent or unusable, turnover is `None` and the response says it is unavailable. It must never be presented as zero merely because no prior allocation was provided.

## Optimization Audit Contract

A successful optimization payload includes:

- `optimization_type`;
- `risk_tolerance`;
- `weights`;
- `expected_annualized_return`;
- `expected_cvar_95`;
- `target_annual_return_floor`;
- `target_daily_return_floor`;
- `target_return_constraint_used`;
- `fallback_applied`;
- `fallback_reason`;
- `solver_name`;
- `solver_status`;
- `objective_value`;
- `max_weight_constraint`;
- `max_observed_weight`;
- `weight_cap_utilization`;
- `hhi`;
- `effective_number_of_holdings`;
- `turnover`;
- `graph_exposure`;
- `instability_index`;
- `lambda_t`;
- `effective_window_start` and `effective_window_end`;
- historical pricing dates and graph scores already returned by the optimizer.

The lightweight governance payload must preserve these scalar audit fields. Large covariance and correlation matrices remain excluded from chatbot context.

## Chat Response

The governance response continues to lead with the target date, tickers, weights, systemic-risk scores, expected return, CVaR, instability, and graph penalty. It then adds a compact optimization audit:

- risk profile;
- solver and status;
- effective historical window;
- maximum-weight cap and observed maximum weight;
- whether the return constraint was relaxed;
- HHI and effective holdings;
- graph exposure;
- turnover when available.

If fallback was applied, the response contains an explicit warning and reason. Percentages use percentage formatting; HHI, instability, lambda, and graph exposure remain decimal scores.

## Logging

Log the selected solver, status, cap, fallback state, and fallback reason. Logs must not include full covariance matrices or confidential user data.

## Testing

Tests are added before production changes and must cover:

1. Annual-to-daily geometric conversion.
2. Successful optimization respects the maximum-weight cap.
3. Return-floor fallback retains the maximum-weight cap.
4. No capped solution returns an error rather than an uncapped portfolio.
5. HHI and effective-holdings calculations.
6. Turnover with aligned prior weights and `None` without prior weights.
7. Lightweight payload retains audit scalars and omits large matrices.
8. Chat markdown renders solver, constraints, concentration, fallback warning, and optional turnover.
9. Existing governance pipeline and chatbot continuity tests remain green.

## Compatibility and Rollout

New optimizer arguments are optional so current tool calls remain valid. Existing payload keys retain their meanings. The implementation stays within the current chatbot governance path and does not alter notebook artifacts or frontend chart contracts.
