import { getBarChartDefinition } from './barChartRegistry.js';
import { chooseLayout, getMainMetric, inferBarMode, shouldShowLabels } from './barChartIntelligence.js';
import { normalizePlotId } from './barChartSchema.js';

const DEFAULT_COLORS = ['#4f63f6', '#ffc857', '#f25467', '#38bdf8', '#4cc98a', '#e879b9', '#fb923c', '#818cf8'];
const FIELD_LABELS = {
  allocation_percent: 'Allocation (%)',
  sector_weight_percent: 'Sector weight (%)',
  hhi_value: 'HHI',
  effective_count: 'Effective count',
  risk_contribution_percent: 'Risk contribution (%)',
  allocation_change_percent: 'Allocation change (percentage points)',
  centrality_score: 'Eigenvector centrality',
  ticker: 'Ticker',
  sector: 'Sector',
  portfolio: 'Portfolio',
  strategy: 'Strategy',
};

export function adaptBarChartPayload(spec) {
  if (!spec) return emptyAdaptedSpec('No chart data');
  const normalized = spec.chart_type === 'bar' ? normalizeStandardPayload(spec) : normalizeLegacyPlotSpec(spec);
  return buildMuiBarSpec(normalized);
}

export function normalizeStandardPayload(payload) {
  const plotId = normalizePlotId(payload.plot_id);
  const definition = getBarChartDefinition(plotId);
  const barMode = payload.bar_mode || definition?.preferredMode || inferBarMode(payload);
  const unit = payload.unit || definition?.unit || 'none';
  const series = (payload.series || []).map((series, index) => ({
    key: series.key,
    label: series.label || series.key || `Series ${index + 1}`,
    color: series.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    stack: series.stack || (barMode === 'stacked' ? 'stack' : undefined),
  }));

  return {
    ...payload,
    plot_id: plotId,
    chart_type: 'bar',
    title: payload.title || definition?.title || 'Bar Chart',
    bar_mode: barMode,
    unit,
    sort: payload.sort || definition?.sort || 'none',
    x_axis: payload.x_axis || definition?.x_axis || 'category',
    y_axis: payload.y_axis || definition?.y_axis || 'value',
    series,
    data: Array.isArray(payload.data) ? [...payload.data] : [],
  };
}

export function normalizeLegacyPlotSpec(spec) {
  const categories = [];
  const seen = new Set();
  spec.series?.forEach((series) => {
    series?.data?.forEach((point) => {
      const label = String(point.x ?? '');
      if (!seen.has(label)) {
        seen.add(label);
        categories.push(label);
      }
    });
  });

  const data = categories.map((category) => {
    const row = { category };
    spec.series?.forEach((series) => {
      const key = series.name || series.label || 'value';
      const point = series?.data?.find((candidate) => String(candidate.x ?? '') === category);
      row[key] = Number(point?.y ?? 0);
    });
    return row;
  });

  const firstKey = spec.series?.[0]?.name || spec.series?.[0]?.label || 'value';
  const series = (spec.series || []).map((seriesItem, index) => ({
    key: seriesItem.name || seriesItem.label || `Series ${index + 1}`,
    label: seriesItem.label || seriesItem.name || `Series ${index + 1}`,
    color: seriesItem.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    stack: seriesItem.stack,
    barLabel: seriesItem.barLabel,
    minBarSize: seriesItem.minBarSize,
    highlightScope: seriesItem.highlightScope,
  }));

  return {
    plot_id: normalizePlotId(spec.plot_id || spec.title || 'legacy_bar_chart'),
    chart_type: 'bar',
    plot_type: 'bar',
    title: spec.title || 'Bar Chart',
    bar_mode: spec.layout === 'horizontal' ? 'horizontal' : inferBarMode({ ...spec, data, series }),
    x_axis: spec.layout === 'horizontal' ? firstKey : 'category',
    y_axis: spec.layout === 'horizontal' ? 'category' : firstKey,
    unit: spec.y_format || spec.unit || 'none',
    sort: spec.sort || 'none',
    series,
    data,
    thresholds: spec.thresholds || [],
    layout: spec.layout,
    xAxis: spec.xAxis,
    yAxis: spec.yAxis,
    grid: spec.grid,
    borderRadius: spec.borderRadius,
    categoryGapRatio: spec.categoryGapRatio,
    barGapRatio: spec.barGapRatio,
    renderer: spec.renderer,
    show_labels: spec.show_labels,
    height: spec.height,
  };
}

function buildMuiBarSpec(payload) {
  const mainMetric = getMainMetric(payload);
  const categoryKey = inferCategoryKey(payload, mainMetric);
  const sortedData = sortAndLimitData(payload.data, payload.sort, mainMetric, payload.top_n);
  const layout = payload.layout || chooseLayout({ ...payload, data: sortedData });
  const categories = sortedData.map((row) => row[categoryKey]);
  const categoryWidth = Math.max(78, Math.min(220, categories.reduce((max, label) => Math.max(max, String(label ?? '').length), 0) * 7 + 28));
  const pointCount = sortedData.length * Math.max(1, payload.series.length);
  const chartHeight = Math.max(260, Math.min(860, payload.height || (layout === 'horizontal' ? sortedData.length * 36 + 116 : 340)));
  const showLabels = shouldShowLabels({ ...payload, data: sortedData });

  const series = payload.series.map((seriesItem, index) => ({
    dataKey: seriesItem.key,
    label: seriesItem.label || fieldLabel(seriesItem.key),
    color: seriesItem.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    valueFormatter: getFormatter(payload.unit),
    ...(seriesItem.stack ? { stack: seriesItem.stack } : {}),
    ...(seriesItem.minBarSize != null ? { minBarSize: seriesItem.minBarSize } : {}),
    ...(seriesItem.highlightScope ? { highlightScope: seriesItem.highlightScope } : { highlightScope: { highlight: 'item', fade: 'global' } }),
    ...(showLabels ? { barLabel: (item) => formatValue(item.value, payload.unit) } : {}),
  }));

  const numericAxis = {
    label: fieldLabel(layout === 'horizontal' ? payload.x_axis : payload.y_axis),
    valueFormatter: getFormatter(payload.unit),
    domainLimit: 'nice',
  };
  const categoryAxis = {
    dataKey: categoryKey,
    data: categories,
    scaleType: 'band',
    tickPlacement: 'middle',
    tickLabelPlacement: 'middle',
    ...(payload.categoryGapRatio != null ? { categoryGapRatio: payload.categoryGapRatio } : { categoryGapRatio: 0.32 }),
    ...(payload.barGapRatio != null ? { barGapRatio: payload.barGapRatio } : { barGapRatio: 0.12 }),
  };

  return {
    valid: sortedData.length > 0 && series.length > 0,
    reason: sortedData.length ? '' : 'No rows available for this bar chart.',
    payload,
    dataset: sortedData,
    series,
    layout,
    renderer: payload.renderer,
    categories,
    chartHeight,
    margin: layout === 'horizontal'
      ? { top: 28, right: 36, left: categoryWidth + 12, bottom: 46 }
      : { top: 28, right: 24, left: 64, bottom: categories.length > 6 ? 88 : 58 },
    xAxis: layout === 'horizontal' ? [numericAxis] : [{ ...categoryAxis, label: fieldLabel(payload.x_axis) }],
    yAxis: layout === 'horizontal' ? [{ ...categoryAxis, label: fieldLabel(payload.y_axis), width: categoryWidth }] : [numericAxis],
    grid: payload.grid || (layout === 'horizontal' ? { vertical: true } : { horizontal: true }),
    borderRadius: payload.borderRadius ?? 5,
    thresholds: Array.isArray(payload.thresholds) ? payload.thresholds : [],
    warnings: payload.warnings || [],
    interpretation: payload.interpretation,
    pointCount,
  };
}

function fieldLabel(field) {
  const key = String(field || '').trim();
  return FIELD_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) || 'Value';
}

function inferCategoryKey(payload, mainMetric) {
  const candidates = [payload.x_axis, payload.y_axis, 'ticker', 'sector', 'strategy', 'portfolio', 'category', 'label'];
  return candidates.find((key) => key && key !== mainMetric && payload.data?.some((row) => row?.[key] != null)) || 'category';
}

function sortAndLimitData(data, sort, metric, topN) {
  const rows = [...(data || [])];
  if (sort === 'ascending') rows.sort((a, b) => Number(a?.[metric] ?? 0) - Number(b?.[metric] ?? 0));
  if (sort === 'descending') rows.sort((a, b) => Number(b?.[metric] ?? 0) - Number(a?.[metric] ?? 0));
  const limit = Number(topN);
  return Number.isFinite(limit) && limit > 0 ? rows.slice(0, limit) : rows;
}

function getFormatter(unit) {
  return (value) => formatValue(value, unit);
}

function formatValue(value, unit) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '';
  if (unit === 'percent' || unit === '%') return `${numeric.toFixed(Math.abs(numeric) < 10 ? 1 : 0)}%`;
  if (unit === 'currency') return `$${numeric.toLocaleString()}`;
  if (unit === 'ratio' || unit === 'decimal') return numeric.toFixed(3);
  if (unit === 'count') return numeric.toFixed(Number.isInteger(numeric) ? 0 : 1);
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(2);
}

function emptyAdaptedSpec(reason) {
  return { valid: false, reason, dataset: [], series: [], xAxis: [], yAxis: [], chartHeight: 260, margin: {} };
}
