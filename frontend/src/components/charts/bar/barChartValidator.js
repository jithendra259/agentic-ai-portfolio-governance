import { BAR_MODES, BAR_SORTS, BAR_UNITS, REQUIRED_BAR_FIELDS, normalizePlotId } from './barChartSchema.js';
import { getBarChartDefinition } from './barChartRegistry.js';
import { getMuiPremiumChartsEnabled, premiumUnavailableMessage } from '../chartTierConfig.js';

const CHART_TYPES = new Set(['bar', 'rangeBar', 'histogram', 'mirroredBar']);

export function validateBarChartPayload(payload, options = {}) {
  const errors = [];
  const warnings = [];

  if (!payload || typeof payload !== 'object') {
    return { valid: false, errors: ['payload must be an object'], warnings };
  }

  if (options.requestedPlotId && normalizePlotId(options.requestedPlotId) !== normalizePlotId(payload.plot_id)) {
    errors.push(`requested plot_id ${options.requestedPlotId} does not match returned plot_id ${payload.plot_id}`);
  }

  if (options.universe && payload.universe && String(options.universe) !== String(payload.universe)) {
    errors.push(`requested universe ${options.universe} does not match payload universe ${payload.universe}`);
  }

  const definition = getBarChartDefinition(normalizePlotId(payload.plot_id));
  const chartType = payload.chart_type || definition?.chart_type || 'bar';
  const chartTier = payload.chart_tier || definition?.chart_tier || 'free';
  const requiresPremium = Boolean(payload.requires_premium ?? definition?.requires_premium);
  const premiumEnabled = getMuiPremiumChartsEnabled(options);
  const fallbackChart = payload.fallback_chart || definition?.fallback_chart || null;

  REQUIRED_BAR_FIELDS.forEach((field) => {
    if (field === 'bar_mode' && chartType !== 'bar') return;
    if (payload[field] == null || payload[field] === '') errors.push(`missing ${field}`);
  });

  if (!CHART_TYPES.has(chartType)) errors.push(`invalid chart_type ${chartType}`);
  if (payload.bar_mode && !BAR_MODES.has(payload.bar_mode)) errors.push(`invalid bar_mode ${payload.bar_mode}`);
  if (payload.unit && !BAR_UNITS.has(payload.unit)) errors.push(`invalid unit ${payload.unit}`);
  if (payload.sort && !BAR_SORTS.has(payload.sort)) errors.push(`invalid sort ${payload.sort}`);
  if (!Array.isArray(payload.series) || payload.series.length === 0) errors.push('series must contain at least one series');
  if (!Array.isArray(payload.data) || payload.data.length === 0) errors.push('data must contain at least one row');

  if (requiresPremium && !premiumEnabled) {
    if (fallbackChart) {
      warnings.push(premiumUnavailableMessage(fallbackChart));
    } else {
      errors.push(premiumUnavailableMessage(null));
    }
  }

  const requiredFields = payload.required_fields || definition?.required_fields || definition?.requiredFields || [];
  if (Array.isArray(payload.data)) {
    payload.data.forEach((row, index) => {
      requiredFields.forEach((field) => {
        if (row?.[field] == null) errors.push(`row ${index} missing required field ${field}`);
      });
      if (chartType === 'rangeBar') validateRangeRow(row, index, payload, definition, errors);
      if (chartType === 'mirroredBar') validateMirroredRow(row, index, errors);
      if (chartType === 'histogram') validateHistogramRow(row, index, errors);
      if (chartType === 'bar') {
        payload.series?.forEach((series) => {
          if (series?.key && row?.[series.key] == null) errors.push(`row ${index} missing series field ${series.key}`);
        });
      }
    });
  }

  if (payload.unit === 'percent') {
    const percentValues = [];
    payload.data?.forEach((row) => {
      payload.series?.forEach((series) => {
        const value = Number(row?.[series.key]);
        if (Number.isFinite(value)) percentValues.push(value);
      });
    });
    if (percentValues.some((value) => Math.abs(value) > 0 && Math.abs(value) <= 1)) {
      warnings.push('percentage values look like fractions; UI expects percentage points such as 15 for 15%');
    }
  }

  validateConcentrationMath(payload, errors);

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    chart_tier: chartTier,
    requires_premium: requiresPremium,
    premium_enabled: premiumEnabled,
    fallback_chart: fallbackChart,
    premium_unavailable: requiresPremium && !premiumEnabled,
  };
}

function validateRangeRow(row, index, payload, definition, errors) {
  const startField = payload.rangeStartField || definition?.rangeStartField || (row?.start != null ? 'start' : 'start_value');
  const endField = payload.rangeEndField || definition?.rangeEndField || (row?.end != null ? 'end' : 'end_value');
  if (row?.[startField] == null) errors.push(`row ${index} missing required field ${startField}`);
  if (row?.[endField] == null) errors.push(`row ${index} missing required field ${endField}`);
}

function validateMirroredRow(row, index, errors) {
  if (row?.current_weight == null) errors.push(`row ${index} missing required field current_weight`);
  if (row?.advisory_weight == null) errors.push(`row ${index} missing required field advisory_weight`);
}

function validateHistogramRow(row, index, errors) {
  for (const field of ['bin_start', 'bin_end', 'count']) {
    if (row?.[field] == null) errors.push(`row ${index} missing required field ${field}`);
  }
}

function validateConcentrationMath(payload, errors) {
  const metrics = payload.metrics || payload;
  const tickerHhi = Number(metrics.ticker_hhi);
  const tickerEffective = Number(metrics.ticker_effective_holdings);
  const sectorHhi = Number(metrics.sector_hhi);
  const sectorEffective = Number(metrics.sector_effective_sectors);

  if (Number.isFinite(tickerHhi) && Number.isFinite(tickerEffective)) {
    assertClose(tickerEffective, 1 / tickerHhi, 'ticker_effective_holdings must equal 1 / ticker_hhi', errors);
  }
  if (Number.isFinite(sectorHhi) && Number.isFinite(sectorEffective)) {
    assertClose(sectorEffective, 1 / sectorHhi, 'sector_effective_sectors must equal 1 / sector_hhi', errors);
  }
}

function assertClose(actual, expected, message, errors) {
  if (!Number.isFinite(expected)) return;
  if (Math.abs(actual - expected) > 1e-6) errors.push(message);
}





