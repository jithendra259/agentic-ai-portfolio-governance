import { getMuiPremiumChartsEnabled, premiumUnavailableMessage } from './chartTierConfig.js';
import { getScatterChartDefinition, SCATTER_CHART_TYPES } from './scatterChartRegistry.js';

export function validateScatterChartPayload(payload, options = {}) {
  const errors = [];
  const warnings = [];

  if (!payload || typeof payload !== 'object') {
    return { valid: false, errors: ['payload must be an object'], warnings };
  }

  const definition = getScatterChartDefinition(payload.plot_id);
  const chartType = payload.chart_type || definition?.chart_type || 'scatter';
  const xField = payload.x_axis || definition?.x_axis;
  const yField = payload.y_axis || definition?.y_axis;
  const pointId = payload.point_id || definition?.point_id || 'id';
  const colorAxis = payload.color_axis || definition?.color_axis;
  const sizeAxis = payload.size_axis || definition?.size_axis;
  const data = Array.isArray(payload.data) ? payload.data : [];
  const requiredFields = payload.required_fields || definition?.required_fields || [];

  if (!SCATTER_CHART_TYPES.has(chartType)) errors.push(`invalid chart_type ${chartType}`);
  if (!xField) errors.push('missing x_axis');
  if (!yField) errors.push('missing y_axis');
  if (!pointId) errors.push('missing point_id');
  if (!data.length) errors.push('data must contain at least two points');
  if (!Array.isArray(payload.series) || payload.series.length === 0) errors.push('series must contain at least one series');

  let validPointCount = 0;
  data.forEach((row, index) => {
    requiredFields.forEach((field) => {
      if (row?.[field] == null) errors.push(`row ${index} missing required field ${field}`);
    });
    if (xField && row?.[xField] == null) errors.push(`row ${index} missing x field ${xField}`);
    if (yField && row?.[yField] == null) errors.push(`row ${index} missing y field ${yField}`);
    if (pointId && row?.[pointId] == null) errors.push(`row ${index} missing point id ${pointId}`);
    if (colorAxis && row?.[colorAxis] == null) errors.push(`row ${index} missing color field ${colorAxis}`);
    if (sizeAxis && row?.[sizeAxis] == null) errors.push(`row ${index} missing size field ${sizeAxis}`);

    const xValue = Number(row?.[xField]);
    const yValue = Number(row?.[yField]);
    if (Number.isFinite(xValue) && Number.isFinite(yValue)) validPointCount += 1;
    else if (row?.[xField] != null || row?.[yField] != null) errors.push(`row ${index} x/y must be finite numbers`);

    if (sizeAxis && row?.[sizeAxis] != null) {
      const sizeValue = Number(row[sizeAxis]);
      if (!Number.isFinite(sizeValue) || sizeValue < 0) errors.push('bubble size values must be finite and non-negative');
    }
  });

  if (validPointCount < 2) errors.push('scatter plot requires at least two valid points');
  if (chartType === 'scatter_regression' && validPointCount < 3) errors.push('regression scatter requires at least three valid points');
  if (chartType === 'scatter_regression' && payload.regression_used && !payload.regression_line) {
    errors.push('regression scatter requires regression_line metadata');
  }
  if (chartType === 'bubble_scatter' && !sizeAxis) errors.push('bubble scatter requires size_axis');
  if (chartType === 'bubble_scatter' && sizeAxis && !data.some((row) => Number(row?.[sizeAxis]) > 0)) {
    errors.push('bubble scatter requires at least one positive bubble size');
  }
  if (payload.plot_id === 'ownership_overlap_correlation_scatter' && !payload.graph_data_available) {
    errors.push('ownership-overlap scatter requires institutional graph data');
  }
  if (payload.plot_id === 'beta_return_scatter' && !payload.benchmark_available) {
    errors.push('beta-return scatter requires benchmark series');
  }
  if (payload.optimizer_called) errors.push('optimizer must not be called for scatter diagnostics');
  if (payload.advisory_allocation_generated) errors.push('advisory allocation must not be generated for scatter diagnostics');
  if (payload.point_count != null && Number(payload.point_count) !== validPointCount) {
    errors.push('point_count must equal the number of valid points');
  }

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
