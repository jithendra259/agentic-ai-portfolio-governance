import { CHART_TIER } from './chartTierConfig.js';

export const PIE_CHART_TYPES = new Set(['pie', 'donut', 'center_label_donut', 'nested_donut', 'semi_donut']);

function definePieChart(config) {
  const requiredFields = config.required_fields || [];
  return {
    plot_id: config.plot_id,
    chart_tier: config.chart_tier || CHART_TIER.FREE,
    component: config.component || 'PieChart',
    chart_type: config.chart_type || 'donut',
    requires_premium: Boolean(config.requires_premium),
    required_fields: requiredFields,
    required_context: config.required_context || [],
    fallback_chart: config.fallback_chart || null,
    fallback_computation: config.fallback_computation || null,
    allowed_intents: config.allowed_intents || [],
    blocked_intents: config.blocked_intents || ['time_series', 'advisory_allocation', 'optimization'],
    category_field: config.category_field,
    value_field: config.value_field,
    unit: config.unit || 'percent',
    max_slices: config.max_slices,
    max_inner_slices: config.max_inner_slices,
    max_outer_slices: config.max_outer_slices,
    ...config,
  };
}

export const PIE_CHART_REGISTRY = {
  sector_allocation_donut: definePieChart({
    plot_id: 'sector_allocation_donut',
    title: 'Sector Allocation Donut',
    chart_type: 'donut',
    required_fields: ['sector', 'weight_percent'],
    required_context: ['sector_weights'],
    fallback_computation: 'compute_sector_weights_from_ticker_weights',
    fallback_chart: 'sector_allocation_bar',
    category_field: 'sector',
    value_field: 'weight_percent',
    max_slices: 8,
  }),
  ticker_allocation_donut: definePieChart({
    plot_id: 'ticker_allocation_donut',
    title: 'Ticker Allocation Donut',
    chart_type: 'donut',
    required_fields: ['ticker', 'weight_percent'],
    required_context: ['ticker_weights'],
    fallback_chart: 'ticker_concentration_bar',
    category_field: 'ticker',
    value_field: 'weight_percent',
    max_slices: 8,
  }),
  risk_contribution_donut: definePieChart({
    plot_id: 'risk_contribution_donut',
    title: 'Risk Contribution Donut',
    chart_type: 'donut',
    required_fields: ['name', 'risk_contribution_percent'],
    required_context: ['weights', 'covariance_matrix'],
    fallback_computation: 'compute_risk_contribution_from_weights_and_covariance',
    fallback_chart: 'risk_contribution_bar',
    category_field: 'name',
    value_field: 'risk_contribution_percent',
    max_slices: 10,
  }),
  sector_ticker_nested_donut: definePieChart({
    plot_id: 'sector_ticker_nested_donut',
    title: 'Sector and Ticker Nested Donut',
    chart_type: 'nested_donut',
    required_fields: ['sector', 'sector_weight_percent', 'ticker', 'ticker_weight_percent'],
    required_context: ['ticker_weights', 'sector_mapping'],
    fallback_computation: 'compute_sector_ticker_nested_donut',
    category_field: 'ticker',
    value_field: 'ticker_weight_percent',
    max_inner_slices: 8,
    max_outer_slices: 25,
  }),
  portfolio_health_donut: definePieChart({
    plot_id: 'portfolio_health_donut',
    title: 'Portfolio Health Donut',
    chart_type: 'center_label_donut',
    required_fields: ['health_component', 'score'],
    required_context: ['ticker_weights', 'sector_mapping'],
    fallback_computation: 'compute_portfolio_health_components',
    category_field: 'health_component',
    value_field: 'score',
    unit: 'score',
    max_slices: 6,
  }),
  semi_donut_risk_gauge: definePieChart({
    plot_id: 'semi_donut_risk_gauge',
    title: 'Portfolio Risk Gauge',
    chart_type: 'semi_donut',
    required_fields: ['health_component', 'score'],
    required_context: ['ticker_weights', 'sector_mapping'],
    fallback_computation: 'compute_portfolio_health_components',
    category_field: 'health_component',
    value_field: 'score',
    unit: 'score',
    max_slices: 6,
  }),
};

export function getPieChartDefinition(plotId) {
  return PIE_CHART_REGISTRY[String(plotId || '').trim()] || null;
}
