# Portfolio Governance And Stock EDA Knowledge Pack

## Project Identity

Who created the SSRN stock EDA paper:
Ishvah Nabimanya and David Owor authored the paper as part of the WorldQuant University Master of Science in Financial Engineering program. The paper lists Professor Ken Abbott as supervisor.

What the SSRN paper studies:
The paper is a case study of selected banking stocks on the S&P 500. It uses exploratory data analysis and forecasting models to compare stock behavior before and after COVID-era market stress.

When the paper was dated:
The paper is dated 19 August 2024.

Where the study context comes from:
The study context is S&P 500 banking stocks, with emphasis on investor decision-making under market uncertainty and post-COVID volatility.

Why the paper matters:
The paper argues that systematic EDA helps investors reduce intuition-driven decisions, understand market behavior, detect volatility changes, and prepare better modeling inputs.

How the paper approaches analysis:
The workflow begins with data understanding and cleaning, continues through EDA, volatility and correlation analysis, stationarity tests, and then evaluates forecasting models such as ARIMA, GARCH, Linear Regression, Decision Trees, Random Forests, and Gradient Boosting.

## Exploratory Data Analysis

What EDA is:
Exploratory Data Analysis is the first structured pass over data. It checks data shape, quality, distributions, relationships, outliers, trends, and patterns before deeper modeling or portfolio decisions.

Why EDA is needed:
EDA prevents weak downstream conclusions. Models can look accurate while using bad assumptions if missing values, outliers, non-stationarity, scale problems, or unstable correlations are ignored.

How EDA works in this project:
The chatbot should inspect historical price coverage, missing values, duplicate dates, stale data, adjusted close trends, normalized price movement, daily returns, return distributions, outliers, correlations, covariance, rolling volatility, cumulative return, and seasonal summaries.

Where EDA data comes from:
The deployed backend reads historical ticker data from MongoDB when available. When a ticker is missing, fallback sample data can be generated, but the answer must clearly mark fallback/sample data.

When to use EDA:
Use EDA before portfolio optimization, backtesting, predictive modeling, G-CVaR analysis, regime classification, or technical forecasting. EDA is the evidence layer that tells whether the data is fit for analysis.

Who uses EDA:
Investors, analysts, portfolio managers, researchers, risk teams, and the governance chatbot use EDA to understand market structure and data reliability.

## Statistics And Data Types

What quantitative data is:
Quantitative data is numeric and supports arithmetic. Stock prices, volume, returns, volatility, drawdown, beta, Sharpe ratio, and CVaR are quantitative.

What qualitative data is:
Qualitative data describes categories. Sector, company, ticker, country, exchange, regime label, model name, and governance status are qualitative.

What nominal data is:
Nominal data is categorical without natural order, such as ticker symbols, company names, sectors, or market names.

What ordinal data is:
Ordinal data is categorical with order, such as risk levels low, medium, high, and critical, or regimes calm, elevated, crisis.

What continuous data is:
Continuous data can vary smoothly over a range. Close price, adjusted close, return, volatility, and market capitalization are commonly continuous.

What discrete data is:
Discrete data is countable. Number of observations, number of missing rows, count of outliers, number of holdings, and number of triggered rules are discrete.

How data type affects analysis:
Numeric data supports descriptive statistics, distributions, regression, correlation, volatility, and risk metrics. Categorical data supports grouping, filtering, counts, sector comparison, and regime summaries.

## Missing Values And Outliers

What missing values mean:
Missing values indicate absent observations. In financial data they can come from market holidays, incomplete vendor data, ticker inception dates, delistings, stale feeds, or database gaps.

How missing values should be handled:
The chatbot should first report missingness, duplicate dates, stale records, and first available date. It should only impute when the method is disclosed, such as forward/backward filling for aligned return windows.

What outliers mean:
Outliers are unusually large or small values relative to a metric distribution. In finance they can reflect real market shocks, earnings events, crises, data errors, or one-off structural breaks.

Why outliers should not be automatically removed:
Financial outliers can carry important information about stress and tail risk. The chatbot should flag and explain them, not silently delete them.

How outliers are detected in the backend:
The full stock EDA path flags rows with absolute z-score greater than 2 by ticker and metric across price, volume, return, and volatility fields.

## Stock EDA Feature Engineering

What OHLCV means:
OHLCV means open, high, low, close, and volume. These fields support candlestick plots, intraday-range volatility proxies, volume trend analysis, and close-to-open return calculations.

What market return means here:
The full stock EDA endpoint computes market return as `(Close - Open) / Close * 100`. This is an intraday close-versus-open movement expressed as a percentage.

What volatility means here:
The full stock EDA endpoint computes volatility as `(High - Low) / Close * 100`. This is a daily trading-range proxy, not annualized standard deviation.

How seasonal summaries work:
The endpoint groups by year, quarter, month, and day of week. It reports observations, average close, average volume, average volatility, total market return, and average market return.

Why skewness matters:
Skewness describes asymmetry. Negative skew suggests more extreme downside moves; positive skew suggests more extreme upside moves.

Why kurtosis matters:
Kurtosis describes tail heaviness. High kurtosis means more extreme observations than a normal-like distribution, which matters for risk and stress analysis.

## Correlation, Covariance, And PCA

What correlation measures:
Correlation measures how two return series move together. High positive correlation reduces diversification benefits.

Why rolling correlation matters:
Correlations often rise during stress. Rolling correlation helps detect periods when assets become more connected and diversification weakens.

What covariance measures:
Covariance captures joint movement in return units. It is scale-dependent and feeds portfolio variance, risk contribution, and optimization.

What covariance drift means:
Covariance drift measures how much the covariance structure changes over time from a baseline window. Large drift suggests a changing market regime.

What PCA/eigenvalue analysis means:
PCA decomposes return covariance into dominant components. Large first eigenvalue concentration can imply a shared market factor is driving many assets.

## Pre-COVID And Post-COVID Analysis

What pre/post-COVID comparison asks:
It compares market behavior before the COVID shock with behavior after or during the recovery. Typical metrics include close price trend, volatility, volume, returns, stationarity, and model performance.

Why COVID matters for stock EDA:
COVID created a sharp market disruption in early 2020 and changed volatility patterns. The paper reports a decline around early 2020 followed by recovery in many banking stocks.

How the chatbot should answer COVID-period questions:
It should separate periods explicitly, compute comparable metrics for each period, and avoid implying causality unless the evidence supports it.

## Forecasting Models

What ARIMA is:
ARIMA is a statistical time-series model for autoregressive and moving-average behavior after differencing. It is often used when a series can be made stationary.

What GARCH is:
GARCH models time-varying volatility and volatility clustering, making it useful for financial returns where calm and turbulent periods alternate.

What ADF tests:
The Augmented Dickey-Fuller test checks whether a time series is stationary. Stationarity is important before applying many classical time-series models.

What Random Forests and Gradient Boosting do:
They are tree-based machine learning models that learn nonlinear relationships. They can be useful in forecasting when engineered features capture relevant market behavior.

When to use modeling:
Use predictive models after EDA, cleaning, stationarity checks, train/test splitting, and feature engineering. Modeling before EDA increases the risk of misleading outputs.

## Portfolio Governance Framework

What G-CVaR means:
G-CVaR is the project’s graph-regularized conditional value-at-risk framework. It combines historical downside risk with structural ownership or contagion information.

How the governance framework works:
The system reads historical data, computes time-series risk and instability, retrieves institutional overlap or graph context, optimizes advisory portfolio weights, validates constraints, and explains the result with audit traceability.

Why HITL exists:
Human-in-the-loop review exists because high-risk regimes, large turnover, concentration breaches, and structural contagion signals should not be treated as automatic actions.

Who reviews HITL outputs:
A human analyst, supervisor, or portfolio decision-maker reviews the advisory output. The system remains read-only and does not execute trades.

Where graph RAG fits:
Graph RAG answers questions about institutional holders, ownership overlap, common holders, contagion structure, and which stocks may be systemically connected.

When to use the methodology knowledge base:
Use it for who, what, when, where, why, and how questions about EDA, statistics, paper methodology, G-CVaR, HITL, RAG, forecasting models, stationarity, volatility, correlation, and data quality.

## Safe Answering Rules

How to answer:
Answer directly first, then add evidence and limitations. Use the retrieved knowledge chunks when available. Do not pretend fallback data is real historical data.

What not to do:
Do not execute trades, guarantee investment outcomes, invent source values, hide missing data, or present sample fallback output as final empirical evidence.

When unsure:
State what is known, what data source was used, and what should be checked next.
