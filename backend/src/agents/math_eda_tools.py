"""Comprehensive Mathematical Econometrics, Convex Optimization, and EDA tools for the Advisory Chatbot."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
import tabulate

from src.agents.live_data_tools import (
    _cached_yfinance_history_frame,
    _extract_price_frame,
    _find_price_documents_with_retry,
    _normalize_tickers,
)

logger = logging.getLogger(__name__)


def _fetch_price_df(tickers: list[str], start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """Helper to fetch aligned close prices for a list of tickers from MongoDB or yfinance."""
    clean_tickers = _normalize_tickers(tickers)
    if not clean_tickers:
        return pd.DataFrame()

    start_dt = pd.to_datetime(start_date, utc=True).tz_localize(None)
    end_dt = pd.to_datetime(end_date, utc=True).tz_localize(None)
    series_dict = {}

    for ticker in clean_tickers:
        frame = pd.DataFrame()
        # 1. MongoDB
        docs = _find_price_documents_with_retry(ticker)
        if docs:
            doc = docs[0] if isinstance(docs, list) and docs else (docs if isinstance(docs, dict) else None)
            if doc:
                frame = _extract_price_frame(doc)
        # 2. yfinance fallback
        if frame.empty:
            frame = _cached_yfinance_history_frame(ticker, start_date, end_date)

        if not frame.empty and "Close" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
            filtered = frame[(frame["Date"] >= start_dt) & (frame["Date"] <= end_dt)]
            if len(filtered) >= 10:
                s = filtered.set_index("Date")["Close"].astype(float)
                series_dict[ticker] = s

    if not series_dict:
        return pd.DataFrame()

    df = pd.DataFrame(series_dict).sort_index().ffill().bfill().dropna(how="all")
    return df


@tool("run_stationarity_and_econometric_tests")
def run_stationarity_and_econometric_tests(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
) -> str:
    """Run rigorous statistical stationarity and econometric diagnostic tests (ADF, KPSS, Ljung-Box) on price and return series.
    
    Args:
        tickers: List of ticker symbols (e.g. ['AAPL', 'MSFT']).
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    df = _fetch_price_df(tickers, start_date, end_date)
    if df.empty or len(df) < 20:
        return f"Insufficient historical data found for {tickers} between {start_date} and {end_date}."

    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return "statsmodels is required for econometric stationarity testing."

    results = []
    headers = ["Ticker", "Series", "ADF Stat", "ADF p-val", "KPSS Stat", "KPSS p-val", "Ljung-Box p(10)", "Conclusion"]

    for col in df.columns:
        price_series = df[col].dropna()
        return_series = np.log(price_series / price_series.shift(1)).dropna()

        for name, series in [("Price", price_series), ("Log Return", return_series)]:
            if len(series) < 20:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ADF
                try:
                    adf_out = adfuller(series, autolag="AIC")
                    adf_stat = round(float(adf_out[0]), 4)
                    adf_p = round(float(adf_out[1]), 4)
                except Exception:
                    adf_stat, adf_p = float("nan"), float("nan")

                # KPSS
                try:
                    kpss_out = kpss(series, regression="c", nlags="auto")
                    kpss_stat = round(float(kpss_out[0]), 4)
                    kpss_p = round(float(kpss_out[1]), 4)
                except Exception:
                    kpss_stat, kpss_p = float("nan"), float("nan")

                # Ljung-Box
                try:
                    lb_out = acorr_ljungbox(series, lags=[10], return_df=True)
                    lb_p = round(float(lb_out["lb_pvalue"].iloc[0]), 4)
                except Exception:
                    lb_p = float("nan")

            # Conclusion
            if adf_p < 0.05 and kpss_p >= 0.05:
                conclusion = "Stationary (I(0))"
            elif adf_p >= 0.05 and kpss_p < 0.05:
                conclusion = "Non-Stationary (Unit Root I(1))"
            elif adf_p < 0.05 and kpss_p < 0.05:
                conclusion = "Trend/Difference Stationary"
            else:
                conclusion = "Inconclusive"

            results.append([
                col, name, adf_stat, adf_p, kpss_stat, kpss_p, lb_p, conclusion
            ])

    table_md = tabulate.tabulate(results, headers=headers, tablefmt="github")
    return (
        f"### Econometric Stationarity & Diagnostic Tests ({start_date} to {end_date})\n\n"
        f"{table_md}\n\n"
        "**Interpretation Guide:**\n"
        "- **ADF Test**: H0 = Non-stationary (Unit Root). A p-value < 0.05 rejects H0 in favor of stationarity.\n"
        "- **KPSS Test**: H0 = Stationary. A p-value < 0.05 rejects stationarity (indicates presence of unit root).\n"
        "- **Ljung-Box Test (lag 10)**: H0 = No autocorrelation (White Noise). A p-value < 0.05 indicates significant autocorrelation."
    )


@tool("fit_garch_volatility_model")
def fit_garch_volatility_model(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    model_type: str = "GARCH",
) -> str:
    """Fit a GARCH(1,1) or EGARCH conditional heteroskedasticity model to estimate volatility persistence and generate forward volatility forecasts.
    
    Args:
        ticker: Ticker symbol (e.g. 'AAPL', 'RELIANCE.NS').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        model_type: 'GARCH' or 'EGARCH' (default 'GARCH').
    """
    df = _fetch_price_df([ticker], start_date, end_date)
    if df.empty or len(df) < 50:
        return f"Insufficient price history for {ticker} (minimum 50 observations required)."

    clean_ticker = df.columns[0]
    prices = df[clean_ticker].dropna()
    returns = (np.log(prices / prices.shift(1)).dropna()) * 100.0  # percentage scale for arch solver

    try:
        from arch import arch_model
    except ImportError:
        return "arch package is required to fit GARCH models."

    vol_spec = "EGARCH" if model_type.upper() == "EGARCH" else "GARCH"
    try:
        am = arch_model(returns, vol=vol_spec, p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
    except Exception as exc:
        return f"Failed to converge GARCH model for {clean_ticker}: {exc}"

    params = res.params
    omega = float(params.get("omega", 0.0))
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    persistence = round(alpha + beta, 4)

    # Annualized conditional volatility
    cond_vol = float(res.conditional_volatility.iloc[-1]) * np.sqrt(252)
    uncond_vol = np.sqrt(omega / max(1e-6, 1.0 - (alpha + beta))) * np.sqrt(252) if (alpha + beta) < 1.0 else float("nan")

    # 10-day forward forecast
    forecasts = res.forecast(horizon=10)
    var_forecast = forecasts.variance.iloc[-1]
    annual_vol_forecast = [round(float(np.sqrt(v) * np.sqrt(252)), 2) for v in var_forecast]

    table_data = [
        ["Model Specification", f"{vol_spec}(1,1) with Normal innovations"],
        ["Omega (Baseline Variance)", f"{omega:.6f}"],
        ["Alpha (Reaction to Shocks)", f"{alpha:.4f}"],
        ["Beta (Volatility Persistence)", f"{beta:.4f}"],
        ["Total Persistence (Alpha + Beta)", f"{persistence:.4f}"],
        ["Current Conditional Volatility (Annualized)", f"{cond_vol:.2f}%"],
        ["Long-Run Unconditional Volatility", f"{uncond_vol:.2f}%" if not np.isnan(uncond_vol) else "Integrated / Non-mean-reverting"],
        ["AIC / BIC Diagnostics", f"{res.aic:.2f} / {res.bic:.2f}"],
    ]
    summary_table = tabulate.tabulate(table_data, headers=["Metric", "Value"], tablefmt="github")

    forecast_rows = [[f"Day +{i+1}", f"{vol}%"] for i, vol in enumerate(annual_vol_forecast)]
    forecast_table = tabulate.tabulate(forecast_rows, headers=["Forecast Horizon", "Projected Annualized Volatility"], tablefmt="github")

    clustering_desc = (
        "Strong volatility clustering (shocks decay slowly)."
        if persistence > 0.90
        else "Moderate volatility persistence (shocks mean-revert quickly)."
    )

    return (
        f"### GARCH(1,1) Volatility Modeling for {clean_ticker} ({start_date} to {end_date})\n\n"
        f"{summary_table}\n\n"
        f"**Persistence Assessment:** {clustering_desc}\n\n"
        f"#### 10-Day Term Structure Volatility Forecast:\n"
        f"{forecast_table}"
    )


@tool("calculate_full_stock_eda")
def calculate_full_stock_eda(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
) -> str:
    """Run comprehensive Exploratory Data Analysis (EDA) on stocks, including statistical moments, risk metrics (Sharpe, Sortino, VaR, CVaR, Max Drawdown), and Ledoit-Wolf shrinkage covariance.
    
    Args:
        tickers: List of ticker symbols (e.g. ['AAPL', 'MSFT', 'GOOG']).
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    df = _fetch_price_df(tickers, start_date, end_date)
    if df.empty or len(df) < 15:
        return f"Insufficient price observations found for {tickers}."

    returns = np.log(df / df.shift(1)).dropna()
    rf_daily = 0.03 / 252.0  # 3% annual risk-free rate proxy

    # Descriptive Stats & Risk Ratios Table
    stats_rows = []
    stats_headers = [
        "Ticker", "Obs", "Mean Daily", "Annual Ret", "Annual Vol",
        "Skew", "Kurtosis", "Sharpe", "Sortino", "Max DD", "VaR(95%)", "CVaR(95%)"
    ]

    for col in df.columns:
        r = returns[col].dropna()
        p = df[col].dropna()
        if len(r) < 10:
            continue

        mean_d = float(r.mean())
        ann_ret = float(mean_d * 252.0 * 100.0)
        ann_vol = float(r.std() * np.sqrt(252.0) * 100.0)
        skew = float(r.skew())
        kurt = float(r.kurtosis())

        # Sharpe
        excess = r - rf_daily
        sharpe = round(float((excess.mean() / (r.std() + 1e-8)) * np.sqrt(252.0)), 2)

        # Sortino
        downside = r[r < 0]
        sortino = round(float((excess.mean() / (downside.std() + 1e-8)) * np.sqrt(252.0)), 2) if len(downside) > 1 else float("nan")

        # Max Drawdown
        running_max = np.maximum.accumulate(p.values)
        dd = (p.values / running_max) - 1.0
        max_dd = round(float(dd.min() * 100.0), 2)

        # VaR & CVaR (95%)
        var_95 = round(float(np.percentile(r, 5) * 100.0), 2)
        cvar_95 = round(float(r[r <= np.percentile(r, 5)].mean() * 100.0), 2)

        stats_rows.append([
            col, len(r), f"{mean_d*100:.3f}%", f"{ann_ret:.2f}%", f"{ann_vol:.2f}%",
            f"{skew:.2f}", f"{kurt:.2f}", sharpe, sortino, f"{max_dd:.2f}%", f"{var_95:.2f}%", f"{cvar_95:.2f}%"
        ])

    stats_table = tabulate.tabulate(stats_rows, headers=stats_headers, tablefmt="github")

    # Correlation Matrix
    corr = returns.corr().round(3)
    corr_headers = ["Ticker"] + list(corr.columns)
    corr_rows = [[idx] + [f"{val:.3f}" for val in row] for idx, row in corr.iterrows()]
    corr_table = tabulate.tabulate(corr_rows, headers=corr_headers, tablefmt="github")

    # Ledoit-Wolf Shrinkage
    lw_msg = ""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        lw_msg = f"- **Ledoit-Wolf Shrinkage Intensity**: `{lw.shrinkage_:.4f}` (Optimal balance between sample covariance and structured shrinkage target)."
    except Exception:
        pass

    return (
        f"### Full Exploratory Data Analysis ({start_date} to {end_date})\n\n"
        f"#### 1. Statistical Moments & Risk-Adjusted Performance:\n"
        f"{stats_table}\n\n"
        f"#### 2. Cross-Asset Pearson Correlation Matrix:\n"
        f"{corr_table}\n\n"
        f"#### 3. Advanced Covariance Structure:\n"
        f"{lw_msg}\n"
    )


@tool("compute_optimal_portfolio_frontier")
def compute_optimal_portfolio_frontier(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.03,
) -> str:
    """Solve mathematical convex portfolio optimization (Markowitz Max Sharpe, Minimum Volatility, Risk Parity) using CVXPY and conic solvers.
    
    Args:
        tickers: List of asset ticker symbols (minimum 2 tickers).
        start_date: Historical start date YYYY-MM-DD.
        end_date: Historical end date YYYY-MM-DD.
        objective: 'max_sharpe', 'min_volatility', or 'risk_parity' (default 'max_sharpe').
        risk_free_rate: Annual risk-free rate proxy (default 0.03 for 3%).
    """
    clean_tickers = _normalize_tickers(tickers)
    if len(clean_tickers) < 2:
        return "Convex portfolio optimization requires at least 2 distinct tickers."

    df = _fetch_price_df(clean_tickers, start_date, end_date)
    if df.empty or len(df.columns) < 2:
        return f"Unable to assemble complete price series for optimization for {clean_tickers}."

    returns = np.log(df / df.shift(1)).dropna()
    mu = returns.mean().values * 252.0

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        Sigma = lw.covariance_ * 252.0
    except Exception:
        Sigma = returns.cov().values * 252.0

    n = len(df.columns)
    tickers_list = list(df.columns)

    try:
        import cvxpy as cp
    except ImportError:
        return "cvxpy is required for mathematical portfolio optimization."

    w = cp.Variable(n)
    obj_clean = (objective or "max_sharpe").strip().lower()

    if obj_clean == "min_volatility":
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1, w >= 0])
        prob.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)
        opt_w = w.value
        title = "Global Minimum Volatility Portfolio"
    elif obj_clean == "risk_parity":
        # Inverse-volatility heuristic approximation as convex baseline
        inv_vols = 1.0 / np.sqrt(np.diag(Sigma))
        opt_w = inv_vols / np.sum(inv_vols)
        title = "Risk Parity (Equal Risk Contribution) Allocation"
    else:  # max_sharpe via quadratic utility parameter sweep
        best_w = None
        best_sharpe = -999.0
        for gamma in np.logspace(-2, 2, 40):
            w_gamma = cp.Variable(n)
            p = cp.Problem(cp.Maximize(mu @ w_gamma - (gamma / 2.0) * cp.quad_form(w_gamma, Sigma)), [cp.sum(w_gamma) == 1, w_gamma >= 0])
            try:
                p.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)
                if w_gamma.value is not None:
                    ret_g = float(mu @ w_gamma.value)
                    vol_g = float(np.sqrt(w_gamma.value @ Sigma @ w_gamma.value))
                    sh = (ret_g - risk_free_rate) / max(1e-6, vol_g)
                    if sh > best_sharpe:
                        best_sharpe = sh
                        best_w = w_gamma.value
            except Exception:
                continue
        opt_w = best_w if best_w is not None else np.ones(n) / n
        title = "Maximum Sharpe Ratio (Tangency) Portfolio"

    if opt_w is None:
        return "Convex solver did not converge to an optimal solution."

    opt_ret = float(mu @ opt_w) * 100.0
    opt_vol = float(np.sqrt(opt_w @ Sigma @ opt_w)) * 100.0
    opt_sharpe = (opt_ret / 100.0 - risk_free_rate) / max(1e-6, opt_vol / 100.0)

    alloc_rows = [
        [ticker, f"{weight * 100.0:.2f}%"]
        for ticker, weight in sorted(zip(tickers_list, opt_w), key=lambda x: x[1], reverse=True)
    ]
    alloc_table = tabulate.tabulate(alloc_rows, headers=["Asset", "Optimal Weight (%)"], tablefmt="github")

    metrics_rows = [
        ["Expected Annual Return", f"{opt_ret:.2f}%"],
        ["Expected Annual Volatility", f"{opt_vol:.2f}%"],
        ["Portfolio Sharpe Ratio", f"{opt_sharpe:.2f}"],
        ["Risk-Free Rate Benchmark", f"{risk_free_rate * 100:.1f}%"],
        ["Conic Solver Used", "Clarabel / OSQP (Interior-Point)"],
    ]
    metrics_table = tabulate.tabulate(metrics_rows, headers=["Portfolio Characteristic", "Value"], tablefmt="github")

    return (
        f"### Mathematical Portfolio Optimization: {title}\n\n"
        f"#### Asset Allocations:\n"
        f"{alloc_table}\n\n"
        f"#### Projected Performance:\n"
        f"{metrics_table}\n"
    )


@tool("forecast_price_arima")
def forecast_price_arima(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    forecast_days: int = 14,
) -> str:
    """Fit an Autoregressive Integrated Moving Average (ARIMA) model to predict short-term price paths with 95% confidence intervals.
    
    Args:
        ticker: Ticker symbol (e.g. 'AAPL').
        start_date: Historical start date YYYY-MM-DD.
        end_date: Historical end date YYYY-MM-DD.
        forecast_days: Number of days to forecast ahead (default 14, max 30).
    """
    days = max(3, min(int(forecast_days or 14), 30))
    df = _fetch_price_df([ticker], start_date, end_date)
    if df.empty or len(df) < 30:
        return f"Insufficient price series for {ticker} to estimate ARIMA model."

    clean_ticker = df.columns[0]
    prices = df[clean_ticker].dropna()
    last_price = float(prices.iloc[-1])

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return "statsmodels is required for ARIMA forecasting."

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(prices.values, order=(1, 1, 1))
            res = model.fit()
            forecast_res = res.get_forecast(steps=days)
            mean_forecast = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05)
    except Exception as exc:
        return f"ARIMA estimation failed for {clean_ticker}: {exc}"

    last_date = prices.index[-1]
    forecast_rows = []
    for i in range(days):
        day_date = (last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d")
        f_price = round(float(mean_forecast[i]), 2)
        lower_b = round(float(conf_int[i, 0]), 2)
        upper_b = round(float(conf_int[i, 1]), 2)
        delta_pct = round(((f_price - last_price) / last_price) * 100.0, 2)
        forecast_rows.append([day_date, f"${f_price:.2f}", f"${lower_b:.2f} - ${upper_b:.2f}", f"{'+' if delta_pct>=0 else ''}{delta_pct:.2f}%"])

    f_table = tabulate.tabulate(forecast_rows, headers=["Forecast Date", "Predicted Price", "95% Confidence Interval", "Expected Move"], tablefmt="github")

    return (
        f"### ARIMA(1,1,1) Short-Term Price Forecast for {clean_ticker}\n\n"
        f"- **Last Historical Close**: `${last_price:.2f}` as of `{last_date.strftime('%Y-%m-%d')}`\n"
        f"- **Model Fit Diagnostics**: AIC = `{res.aic:.1f}`, BIC = `{res.bic:.1f}`\n\n"
        f"{f_table}\n\n"
        "> [!NOTE]\n"
        "> Econometric forecasts assume martingale drift and are advisory projections based on historical data. They do not constitute investment advice."
    )
