import { CHART_TIER } from '../chartTierConfig.js';

function defineLineChart(config) {
  const requiredFields = config.required_fields || [];
  const optionalFields = config.optional_fields || [];
  const fallbackChart = config.fallback_chart || null;
  return {
    plot_id: config.plot_id,
    chart_type: config.chart_type || 'line',
    chart_tier: config.chart_tier || CHART_TIER.FREE,
    component: config.component || 'LineChart',
    requires_premium: Boolean(config.requires_premium),
    required_fields: requiredFields,
    optional_fields: optionalFields,
    required_context: config.required_context || [],
    fallback_computation: config.fallback_computation || null,
    allowed_intents: config.allowed_intents || [],
    blocked_intents: config.blocked_intents || [],
    fallback_chart: fallbackChart,
    x_axis: config.x_axis || 'date',
    y_axis: config.y_axis || 'value',
    unit: config.unit || 'none',
    connect_nulls: config.connect_nulls ?? false,
    curve: config.curve || 'linear',
    ...config,
  };
}

export const LINE_CHART_REGISTRY = {
  historical_adjusted_close: defineLineChart({
    plot_id: 'historical_adjusted_close',
    title: 'Historical Adjusted Close',
    required_fields: ['date', 'adjusted_close'],
    required_context: ['ticker'],
    fallback_computation: 'fetch_adjusted_close_history',
    allowed_intents: ['plot_only', 'data_exploration'],
    blocked_intents: ['advisory_allocation', 'optimization'],
    y_axis: 'adjusted_close',
    unit: 'price',
  }),
  normalized_price_comparison: defineLineChart({
    plot_id: 'normalized_price_comparison',
    title: 'Normalized Price Comparison',
    required_fields: ['date', 'ticker', 'normalized_price'],
    required_context: ['tickers'],
    fallback_computation: 'normalize_adjusted_close_to_100',
    allowed_intents: ['plot_only', 'data_exploration'],
    blocked_intents: ['advisory_allocation', 'optimization'],
    y_axis: 'normalized_price',
    unit: 'index',
  }),
  portfolio_value_over_time: defineLineChart({
    plot_id: 'portfolio_value_over_time',
    title: 'Portfolio Value Over Time',
    required_fields: ['date', 'portfolio_value'],
    required_context: ['weights', 'initial_capital'],
    fallback_computation: 'compute_portfolio_value_from_weights',
    blocked_intents: ['data_quality_only'],
    y_axis: 'portfolio_value',
    unit: 'currency',
  }),
  drawdown_over_time: defineLineChart({
    plot_id: 'drawdown_over_time',
    title: 'Drawdown Over Time',
    chart_type: 'line_area',
    required_fields: ['date', 'drawdown_percent'],
    required_context: ['weights'],
    fallback_computation: 'compute_drawdown_from_portfolio_value',
    y_axis: 'drawdown_percent',
    unit: 'percent',
  }),
  rolling_volatility_over_time: defineLineChart({
    plot_id: 'rolling_volatility_over_time',
    title: 'Rolling Volatility Over Time',
    required_fields: ['date', 'rolling_volatility_percent'],
    fallback_computation: 'compute_rolling_volatility_from_returns',
    y_axis: 'rolling_volatility_percent',
    unit: 'percent',
  }),
  rolling_correlation_over_time: defineLineChart({
    plot_id: 'rolling_correlation_over_time',
    title: 'Rolling Average Correlation',
    required_fields: ['date', 'average_correlation'],
    fallback_computation: 'compute_rolling_average_correlation',
    y_axis: 'average_correlation',
    unit: 'decimal',
  }),
  instability_index_over_time: defineLineChart({
    plot_id: 'instability_index_over_time',
    title: 'Instability Index Over Time',
    chart_tier: CHART_TIER.PREMIUM,
    component: 'ChartsDataProviderPro',
    requires_premium: true,
    required_fields: ['date', 'instability_index'],
    optional_fields: ['calm_threshold', 'crisis_threshold', 'stress_bands'],
    fallback_chart: 'basic_instability_line',
    fallback_computation: 'compute_instability_index_if_components_available',
    allowed_intents: ['instability_regime', 'plot_only'],
    blocked_intents: ['advisory_allocation', 'optimization'],
    y_axis: 'instability_index',
    unit: 'decimal',
  }),
  cvar_over_time: defineLineChart({
    plot_id: 'cvar_over_time',
    title: 'CVaR Over Time',
    required_fields: ['date', 'cvar_95'],
    required_context: ['weights'],
    fallback_computation: 'compute_rolling_var_cvar',
    y_axis: 'cvar_95',
    unit: 'percent',
  }),
  governance_metric_threshold_over_time: defineLineChart({
    plot_id: 'governance_metric_threshold_over_time',
    title: 'Governance Metric Threshold Over Time',
    required_fields: ['date', 'metric_value'],
    optional_fields: ['threshold_value', 'threshold_label'],
    fallback_computation: 'compute_governance_metric_threshold_series',
    allowed_intents: ['plot_only', 'risk_governance', 'instability_regime'],
    blocked_intents: ['advisory_allocation', 'optimization'],
    y_axis: 'metric_value',
    unit: 'decimal',
  }),
};

export function getLineChartDefinition(plotId) {
  return LINE_CHART_REGISTRY[String(plotId || '').trim()] || null;
}


