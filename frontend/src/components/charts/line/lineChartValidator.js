import { getMuiPremiumChartsEnabled, premiumUnavailableMessage } from '../chartTierConfig.js';
import { getLineChartDefinition } from './lineChartRegistry.js';

const LINE_TYPES = new Set(['line', 'line_area', 'stacked_area', 'dual_axis_line']);

export function validateLineChartPayload(payload, options = {}) {
  const errors = [];
  const warnings = [];

  if (!payload || typeof payload !== 'object') {
    return { valid: false, errors: ['payload must be an object'], warnings };
  }

  const definition = getLineChartDefinition(payload.plot_id);
  const chartType = payload.chart_type || definition?.chart_type || 'line';
  const requiredFields = payload.required_fields || definition?.required_fields || [];
  const data = Array.isArray(payload.data) ? payload.data : [];

  if (!LINE_TYPES.has(chartType)) errors.push(`invalid chart_type ${chartType}`);
  if (!data.length) errors.push('data must contain at least one row');
  if (!Array.isArray(payload.series) || payload.series.length === 0) errors.push('series must contain at least one series');

  requiredFields.forEach((field) => {
    data.forEach((row, index) => {
      if (row?.[field] == null && field !== payload.y_axis) {
        errors.push(`row ${index} missing required field ${field}`);
      }
    });
  });

  if (data.length) {
    const dates = data.map((row) => parseDate(row?.date));
    if (dates.some((date) => date == null)) errors.push('line chart dates must parse as dates');
    const validDates = dates.filter(Boolean);
    if (validDates.length && !isAscending(validDates)) errors.push('line chart dates must be sorted ascending');
  }

  const yField = payload.y_axis || definition?.y_axis;
  if (!yField) {
    errors.push('missing y_axis');
  } else if (data.length && data.every((row) => row?.[yField] == null || !Number.isFinite(Number(row?.[yField])))) {
    errors.push('line chart y-values are all null or non-numeric');
  }

  const tickers = payload.tickers_used || payload.tickers || [];
  if (!tickers.length) errors.push('ticker list is empty');
  if (payload.plot_id === 'historical_adjusted_close' && tickers.length > 1) {
    errors.push('raw multi-ticker price comparison must use normalized_price_comparison');
  }
  if (payload.optimizer_called) errors.push('optimizer must not be called for line-only plots');
  if (payload.advisory_allocation_generated) errors.push('advisory allocation must not be generated for line-only plots');
  if (payload.connect_nulls !== false && payload.connectNulls !== false) {
    warnings.push('connectNulls should default to false for financial line charts');
  }

  const requiresPremium = Boolean(payload.requires_premium ?? definition?.requires_premium);
  const fallbackChart = payload.fallback_chart || definition?.fallback_chart || null;
  const premiumEnabled = getMuiPremiumChartsEnabled(options);
  if (requiresPremium && !premiumEnabled) {
    if (fallbackChart) {
      warnings.push(premiumUnavailableMessage(fallbackChart));
    } else {
      errors.push(premiumUnavailableMessage(null));
    }
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

function parseDate(value) {
  if (value == null) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function isAscending(dates) {
  return dates.every((date, index) => index === 0 || date.getTime() >= dates[index - 1].getTime());
}



