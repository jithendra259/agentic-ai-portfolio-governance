import { getMuiPremiumChartsEnabled, premiumUnavailableMessage } from '../chartTierConfig.js';
import { getPieChartDefinition, PIE_CHART_TYPES } from './pieChartRegistry.js';

const PERCENT_SUM_TOLERANCE = 0.5;

export function validatePieChartPayload(payload, options = {}) {
  const errors = [];
  const warnings = [];

  if (!payload || typeof payload !== 'object') {
    return { valid: false, errors: ['payload must be an object'], warnings };
  }

  const definition = getPieChartDefinition(payload.plot_id);
  const chartType = payload.chart_type || definition?.chart_type || 'donut';
  const categoryField = payload.category_field || definition?.category_field || payload.x_axis;
  const valueField = payload.value_field || definition?.value_field || payload.y_axis;
  const data = Array.isArray(payload.data) ? payload.data : [];
  const requiredFields = payload.required_fields || definition?.required_fields || [];

  if (!PIE_CHART_TYPES.has(chartType)) errors.push(`invalid chart_type ${chartType}`);
  if (!categoryField) errors.push('missing category_field');
  if (!valueField) errors.push('missing value_field');
  if (!data.length) errors.push('data must contain at least one slice');
  if (!Array.isArray(payload.series) || payload.series.length === 0) errors.push('series must contain at least one series');

  data.forEach((row, index) => {
    requiredFields.forEach((field) => {
      if (row?.[field] == null) errors.push(`row ${index} missing required field ${field}`);
    });
    if (categoryField && row?.[categoryField] == null) errors.push(`row ${index} missing category field ${categoryField}`);
    if (valueField && row?.[valueField] == null) errors.push(`row ${index} missing value field ${valueField}`);
    const value = Number(row?.[valueField]);
    if (!Number.isFinite(value)) errors.push(`row ${index} value must be numeric`);
    if (Number.isFinite(value) && value < 0) errors.push('pie/donut values cannot be negative');
  });

  const totalValue = Number(payload.total_value ?? data.reduce((sum, row) => sum + Number(row?.[valueField] || 0), 0));
  if (!Number.isFinite(totalValue) || totalValue <= 0) errors.push('total_value must be greater than zero');

  const sliceCount = Number(payload.slice_count ?? data.length);
  const maxSlices = definition?.max_slices ?? payload.max_slices;
  if (maxSlices && sliceCount > Number(maxSlices) && !payload.explicit_large_pie) {
    errors.push(`slice_count exceeds max_slices ${maxSlices}`);
  }

  if (chartType === 'nested_donut') {
    if (!data.every((row) => row?.sector != null && row?.ticker != null)) {
      errors.push('nested donut requires sector and ticker parent-child fields');
    }
    if (!data.every((row) => row?.sector_weight_percent != null && row?.ticker_weight_percent != null)) {
      errors.push('nested donut requires sector and ticker weight fields');
    }
    if ((payload.series || []).length < 2) errors.push('nested donut requires two series rings');
    const maxOuterSlices = definition?.max_outer_slices ?? payload.max_outer_slices;
    if (maxOuterSlices && data.length > Number(maxOuterSlices)) {
      errors.push(`outer slice count exceeds max_outer_slices ${maxOuterSlices}`);
    }
  }

  if (payload.plot_id === 'portfolio_health_donut' && !payload.metrics?.sector_hhi) {
    errors.push('portfolio health donut requires sector concentration metrics');
  }

  if (payload.time_series_requested || categoryField === 'date') {
    errors.push('pie/donut charts are blocked for time-series data');
  }

  if (isPercentPie(payload, valueField)) {
    const percentTotal = data.reduce((sum, row) => sum + Number(row?.[valueField] || 0), 0);
    if (Math.abs(percentTotal - 100) > PERCENT_SUM_TOLERANCE) {
      errors.push(`percentage values must sum close to 100%; got ${percentTotal.toFixed(4)}`);
    }
  }

  if (payload.optimizer_called) errors.push('optimizer must not be called for pie-only plots');
  if (payload.advisory_allocation_generated) errors.push('advisory allocation must not be generated for pie-only plots');

  const requiresPremium = Boolean(payload.requires_premium ?? definition?.requires_premium);
  const fallbackChart = payload.fallback_chart || definition?.fallback_chart || null;
  const premiumEnabled = getMuiPremiumChartsEnabled(options);
  if (requiresPremium && !premiumEnabled) {
    if (fallbackChart) warnings.push(premiumUnavailableMessage(fallbackChart));
    else errors.push(premiumUnavailableMessage(null));
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    chart_tier: payload.chart_tier || definition?.chart_tier || 'free',
    requires_premium: requiresPremium,
    premium_enabled: premiumEnabled,
    fallback_chart: fallbackChart,
    premium_unavailable: requiresPremium && !premiumEnabled,
  };
}

function isPercentPie(payload, valueField) {
  if (payload.unit === '%' || payload.unit === 'percent') return true;
  if (String(valueField || '').endsWith('_percent')) return true;
  return new Set([
    'sector_allocation_donut',
    'ticker_allocation_donut',
    'risk_contribution_donut',
    'sector_ticker_nested_donut',
  ]).has(String(payload.plot_id || ''));
}




