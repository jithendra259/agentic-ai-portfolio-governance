import { CHART_TIER } from '../chartTierConfig.js';

function defineChart(config) {
  const requiredFields = config.required_fields || config.requiredFields || [];
  const fallbackChart = config.fallback_chart || config.fallbackChart || null;
  return {
    chart_tier: config.chart_tier || CHART_TIER.FREE,
    component: config.component || 'BarChart',
    chart_type: config.chart_type || 'bar',
    requires_premium: Boolean(config.requires_premium),
    fallback_chart: fallbackChart,
    fallbackChart,
    required_fields: requiredFields,
    requiredFields,
    allowed_intents: config.allowed_intents || [],
    blocked_intents: config.blocked_intents || [],
    ...config,
    required_fields: requiredFields,
    requiredFields,
    fallback_chart: fallbackChart,
    fallbackChart,
  };
}

function freeChart(config) {
  return defineChart({
    chart_tier: CHART_TIER.FREE,
    component: 'BarChart',
    chart_type: 'bar',
    requires_premium: false,
    ...config,
  });
}

function premiumChart(config) {
  return defineChart({
    chart_tier: CHART_TIER.PREMIUM,
    component: 'BarChartPremium',
    requires_premium: true,
    ...config,
  });
}

export const BAR_CHART_REGISTRY = {
  ticker_concentration_plot: freeChart({
    title: 'Ticker Concentration',
    x_axis: 'allocation_percent',
    y_axis: 'ticker',
    unit: 'percent',
    sort: 'descending',
    preferredMode: 'horizontal',
    required_fields: ['ticker', 'allocation_percent'],
    interpretation: 'Shows whether individual ticker exposures exceed advisory concentration thresholds.',
  }),
  sector_concentration_plot: freeChart({
    title: 'Sector Concentration',
    x_axis: 'allocation_percent',
    y_axis: 'sector',
    unit: 'percent',
    sort: 'descending',
    preferredMode: 'horizontal',
    required_fields: ['sector', 'allocation_percent'],
  }),
  hhi_concentration_index: freeChart({
    title: 'HHI Concentration Index',
    x_axis: 'portfolio',
    y_axis: 'hhi',
    unit: 'ratio',
    required_fields: ['portfolio', 'hhi'],
  }),
  effective_number_of_holdings: freeChart({
    title: 'Effective Number of Holdings',
    x_axis: 'portfolio',
    y_axis: 'effective_holdings',
    unit: 'count',
    required_fields: ['portfolio', 'effective_holdings'],
  }),
  current_vs_advisory_allocation_by_ticker: freeChart({
    title: 'Current vs Advisory Allocation by Ticker',
    x_axis: 'ticker',
    y_axis: 'allocation_percent',
    unit: 'percent',
    preferredMode: 'grouped',
    required_fields: ['ticker', 'current_allocation_percent', 'advisory_allocation_percent'],
  }),
  current_vs_advisory_sector_allocation: freeChart({
    title: 'Current vs Advisory Sector Allocation',
    x_axis: 'sector',
    y_axis: 'allocation_percent',
    unit: 'percent',
    preferredMode: 'grouped',
    required_fields: ['sector', 'current_allocation_percent', 'advisory_allocation_percent'],
  }),
  allocation_change_by_ticker: freeChart({
    title: 'Allocation Change by Ticker',
    x_axis: 'ticker',
    y_axis: 'allocation_change_percent',
    unit: 'percent',
    preferredMode: 'diverging',
    sort: 'descending',
    required_fields: ['ticker', 'allocation_change_percent'],
  }),
  risk_contribution_by_ticker: freeChart({
    title: 'Risk Contribution by Ticker',
    x_axis: 'risk_contribution_percent',
    y_axis: 'ticker',
    unit: 'percent',
    sort: 'descending',
    preferredMode: 'horizontal',
    required_fields: ['ticker', 'risk_contribution_percent'],
  }),
  allocation_vs_risk_contribution: freeChart({
    title: 'Allocation vs Risk Contribution',
    x_axis: 'ticker',
    y_axis: 'percent',
    unit: 'percent',
    preferredMode: 'grouped',
    required_fields: ['ticker', 'allocation_percent', 'risk_contribution_percent'],
  }),
  cvar_comparison: freeChart({
    title: 'CVaR Comparison',
    x_axis: 'strategy',
    y_axis: 'cvar_95',
    unit: 'percent',
    required_fields: ['strategy', 'cvar_95'],
  }),
  maximum_drawdown_comparison: freeChart({
    title: 'Maximum Drawdown Comparison',
    x_axis: 'strategy',
    y_axis: 'maximum_drawdown',
    unit: 'percent',
    required_fields: ['strategy', 'maximum_drawdown'],
  }),
  sharpe_ratio_comparison: freeChart({
    title: 'Sharpe Ratio Comparison',
    x_axis: 'strategy',
    y_axis: 'sharpe_ratio',
    unit: 'ratio',
    required_fields: ['strategy', 'sharpe_ratio'],
  }),
  eigenvector_centrality_by_ticker: freeChart({
    title: 'Eigenvector Centrality by Ticker',
    x_axis: 'centrality',
    y_axis: 'ticker',
    unit: 'decimal',
    preferredMode: 'horizontal',
    sort: 'descending',
    required_fields: ['ticker', 'centrality'],
  }),
  contagion_penalty_score: freeChart({
    title: 'Contagion Penalty Score',
    x_axis: 'ticker',
    y_axis: 'penalty_score',
    unit: 'decimal',
    sort: 'descending',
    required_fields: ['ticker', 'penalty_score'],
  }),
  strategy_performance_comparison: freeChart({
    title: 'Strategy Performance Comparison',
    x_axis: 'strategy',
    y_axis: 'metric_value',
    unit: 'percent',
    preferredMode: 'grouped',
    required_fields: ['strategy'],
  }),
  return_range_by_ticker: premiumChart({
    title: 'Return Range by Ticker',
    chart_type: 'rangeBar',
    x_axis: 'ticker',
    y_axis: 'return_range_percent',
    unit: 'percent',
    rangeStartField: 'min_return',
    rangeEndField: 'max_return',
    required_fields: ['ticker', 'min_return', 'max_return'],
    fallback_chart: 'standard_min_max_bar',
    allowed_intents: ['eda', 'risk_governance', 'plot_request'],
    blocked_intents: ['advisory_allocation'],
  }),
  volatility_range_by_ticker: premiumChart({
    title: 'Volatility Range by Ticker',
    chart_type: 'rangeBar',
    x_axis: 'ticker',
    y_axis: 'volatility_range_percent',
    unit: 'percent',
    rangeStartField: 'min_volatility',
    rangeEndField: 'max_volatility',
    required_fields: ['ticker', 'min_volatility', 'max_volatility'],
    fallback_chart: 'standard_min_max_bar',
    allowed_intents: ['eda', 'risk_governance', 'plot_request'],
    blocked_intents: ['advisory_allocation'],
  }),
  portfolio_return_waterfall: premiumChart({
    title: 'Portfolio Return Contribution Waterfall',
    chart_type: 'rangeBar',
    x_axis: 'component',
    y_axis: 'return_contribution_percent',
    unit: 'percent',
    rangeStartField: 'start_value',
    rangeEndField: 'end_value',
    required_fields: ['component', 'start_value', 'end_value'],
    fallback_chart: 'contribution_bar',
    allowed_intents: ['eda', 'risk_governance', 'plot_request'],
    blocked_intents: ['advisory_allocation'],
  }),
  risk_contribution_waterfall: premiumChart({
    title: 'Risk Contribution Waterfall',
    chart_type: 'rangeBar',
    x_axis: 'component',
    y_axis: 'risk_contribution_percent',
    unit: 'percent',
    rangeStartField: 'start_value',
    rangeEndField: 'end_value',
    required_fields: ['component', 'start_value', 'end_value'],
    fallback_chart: 'contribution_bar',
    allowed_intents: ['risk_governance', 'plot_request'],
    blocked_intents: ['advisory_allocation'],
  }),
  current_vs_advisory_mirrored_bar: premiumChart({
    title: 'Current Exposure vs Advisory Exposure',
    chart_type: 'mirroredBar',
    x_axis: 'weight_percent',
    y_axis: 'ticker',
    unit: 'percent',
    preferredMode: 'horizontal',
    required_fields: ['ticker', 'current_weight', 'advisory_weight'],
    fallback_chart: 'grouped_bar',
    allowed_intents: ['advisory_allocation', 'diversification', 'plot_request'],
    blocked_intents: ['optimizer_unapproved'],
  }),
  return_distribution_histogram: freeChart({
    title: 'Return Distribution Histogram',
    chart_type: 'histogram',
    x_axis: 'return_bin',
    y_axis: 'count',
    unit: 'count',
    preferredMode: 'vertical',
    required_fields: ['bin_start', 'bin_end', 'count'],
    fallback_chart: null,
  }),
};

export function getBarChartDefinition(plotId) {
  return BAR_CHART_REGISTRY[plotId] || null;
}



