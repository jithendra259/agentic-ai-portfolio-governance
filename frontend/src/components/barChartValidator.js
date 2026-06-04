import { BAR_MODES, BAR_SORTS, BAR_UNITS, REQUIRED_BAR_FIELDS, normalizePlotId } from './barChartSchema.js';
import { getBarChartDefinition } from './barChartRegistry.js';

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

  REQUIRED_BAR_FIELDS.forEach((field) => {
    if (payload[field] == null || payload[field] === '') errors.push(`missing ${field}`);
  });

  if (payload.chart_type !== 'bar') errors.push('chart_type must be bar');
  if (payload.bar_mode && !BAR_MODES.has(payload.bar_mode)) errors.push(`invalid bar_mode ${payload.bar_mode}`);
  if (payload.unit && !BAR_UNITS.has(payload.unit)) errors.push(`invalid unit ${payload.unit}`);
  if (payload.sort && !BAR_SORTS.has(payload.sort)) errors.push(`invalid sort ${payload.sort}`);
  if (!Array.isArray(payload.series) || payload.series.length === 0) errors.push('series must contain at least one series');
  if (!Array.isArray(payload.data) || payload.data.length === 0) errors.push('data must contain at least one row');

  const definition = getBarChartDefinition(normalizePlotId(payload.plot_id));
  const requiredFields = payload.required_fields || definition?.requiredFields || [];
  if (Array.isArray(payload.data)) {
    payload.data.forEach((row, index) => {
      requiredFields.forEach((field) => {
        if (row?.[field] == null) errors.push(`row ${index} missing required field ${field}`);
      });
      payload.series?.forEach((series) => {
        if (series?.key && row?.[series.key] == null) errors.push(`row ${index} missing series field ${series.key}`);
      });
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

  return { valid: errors.length === 0, errors, warnings };
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
