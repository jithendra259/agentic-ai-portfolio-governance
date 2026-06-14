import { formatFinancialValue } from './formatters.js';
import { getMetricConfig } from './radarMetricsConfig.js';

export function metricKey(metric) {
  return typeof metric === 'string' ? metric : (metric?.key || metric?.dataKey || metric?.name || metric?.id);
}

export function normalizeMetricValue(rawValue, config) {
  const numeric = Number(rawValue);
  if (!Number.isFinite(numeric)) return 0;
  const min = Number(config.min);
  const max = Number(config.max);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return 0;
  const clamped = Math.max(min, Math.min(numeric, max));
  let normalized = ((clamped - min) / (max - min)) * 100;
  if (!config.higherIsBetter) normalized = 100 - normalized;
  return Number(normalized.toFixed(4));
}

export function normalizeRadarRows(rows, metrics) {
  if (!Array.isArray(rows)) return [];
  const metricList = Array.isArray(metrics) ? metrics : [];
  return rows.map((row) => {
    const normalized = { ...row };
    metricList.forEach((metric) => {
      const key = metricKey(metric);
      if (!key) return;
      const config = getMetricConfig(metric);
      const rawValue = row[key];
      normalized[`${key}_raw`] = rawValue;
      normalized[key] = normalizeMetricValue(rawValue, config);
    });
    return normalized;
  });
}

export function formatRadarRawValue(value, metric) {
  const config = getMetricConfig(metric);
  return formatFinancialValue(value, config?.rawFormat);
}

export function normalizeRadarSeries(series, metrics) {
  if (!Array.isArray(series)) return [];
  const metricList = Array.isArray(metrics) ? metrics : [];
  return series.map((entry) => ({
    ...entry,
    data: (entry.data || []).map((value, index) => {
      const metric = metricList[index];
      const config = getMetricConfig(metric);
      return normalizeMetricValue(value, config);
    }),
    rawData: entry.rawData || entry.data || [],
  }));
}
