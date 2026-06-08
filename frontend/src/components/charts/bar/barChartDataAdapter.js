import { getBarChartDefinition } from './barChartRegistry.js';
import { chooseLayout, getMainMetric, inferBarMode, shouldShowLabels } from './barChartIntelligence.js';
import { normalizePlotId } from './barChartSchema.js';
import { CHART_TIER, getMuiPremiumChartsEnabled, premiumUnavailableMessage } from '../chartTierConfig.js';

const DEFAULT_COLORS = ['#4f63f6', '#ffc857', '#f25467', '#38bdf8', '#4cc98a', '#e879b9', '#fb923c', '#818cf8'];
const POSITIVE_COLOR = '#4cc98a';
const NEGATIVE_COLOR = '#f25467';
const FIELD_LABELS = {
  allocation_percent: 'Allocation (%)',
  sector_weight_percent: 'Sector weight (%)',
  hhi_value: 'HHI',
  effective_count: 'Effective count',
  risk_contribution_percent: 'Risk contribution (%)',
  allocation_change_percent: 'Allocation change (percentage points)',
  return_range_percent: 'Return range (%)',
  volatility_range_percent: 'Volatility range (%)',
  return_contribution_percent: 'Return contribution (%)',
  weight_percent: 'Exposure (%)',
  count: 'Count',
  centrality_score: 'Eigenvector centrality',
  ticker: 'Ticker',
  sector: 'Sector',
  portfolio: 'Portfolio',
  strategy: 'Strategy',
  component: 'Component',
  return_bin: 'Return bin',
};

const STANDARD_CHART_TYPES = new Set(['bar', 'rangeBar', 'histogram', 'mirroredBar']);

export function adaptBarChartPayload(spec, options = {}) {
  if (!spec) return emptyAdaptedSpec('No chart data');
  const normalized = STANDARD_CHART_TYPES.has(spec.chart_type)
    ? normalizeStandardPayload(spec, options)
    : normalizeLegacyPlotSpec(spec, options);
  return buildMuiBarSpec(normalized);
}

export function normalizeStandardPayload(payload, options = {}) {
  const plotId = normalizePlotId(payload.plot_id);
  const definition = getBarChartDefinition(plotId);
  const premiumEnabled = getMuiPremiumChartsEnabled(options);
  const requiresPremium = Boolean(payload.requires_premium ?? definition?.requires_premium);
  const fallbackChart = payload.fallback_chart || definition?.fallback_chart || null;
  const premiumUnavailable = requiresPremium && !premiumEnabled;
  const chartTier = payload.chart_tier || definition?.chart_tier || CHART_TIER.FREE;
  const requestedChartType = payload.chart_type || definition?.chart_type || 'bar';
  const barMode = payload.bar_mode || definition?.preferredMode || inferBarMode(payload);
  const unit = payload.unit || definition?.unit || 'none';
  const requiredFields = payload.required_fields || definition?.required_fields || definition?.requiredFields || [];

  const normalized = {
    ...payload,
    plot_id: plotId,
    plot_type: 'bar',
    chart_type: requestedChartType,
    chart_tier: chartTier,
    component: payload.component || definition?.component || (requiresPremium ? 'BarChartPremium' : 'BarChart'),
    requires_premium: requiresPremium,
    premium_enabled: premiumEnabled,
    fallback_chart: fallbackChart,
    fallback_used: Boolean(payload.fallback_used),
    title: payload.title || definition?.title || 'Bar Chart',
    bar_mode: barMode,
    unit,
    sort: payload.sort || definition?.sort || 'none',
    x_axis: payload.x_axis || definition?.x_axis || 'category',
    y_axis: payload.y_axis || definition?.y_axis || 'value',
    required_fields: requiredFields,
    rangeStartField: payload.rangeStartField || definition?.rangeStartField,
    rangeEndField: payload.rangeEndField || definition?.rangeEndField,
    data: Array.isArray(payload.data) ? [...payload.data] : [],
    series: normalizeSeries(payload, definition, barMode),
    warnings: [...(payload.warnings || [])],
  };

  if (premiumUnavailable && fallbackChart) {
    return applyPremiumFallback(normalized, fallbackChart);
  }

  if (premiumUnavailable) {
    normalized.status = 'premium_unavailable';
    normalized.warnings.push(premiumUnavailableMessage(null));
  }

  return normalized;
}

export function normalizeLegacyPlotSpec(spec, options = {}) {
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

  return normalizeStandardPayload({
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
  }, options);
}

function normalizeSeries(payload, definition, barMode) {
  const inputSeries = payload.series?.length
    ? payload.series
    : defaultSeriesForPayload(payload, definition);
  return inputSeries.map((series, index) => ({
    key: series.key,
    label: series.label || series.key || `Series ${index + 1}`,
    color: series.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    stack: series.stack || (barMode === 'stacked' ? 'stack' : undefined),
    barLabel: series.barLabel,
    minBarSize: series.minBarSize,
    highlightScope: series.highlightScope,
  }));
}

function defaultSeriesForPayload(payload, definition) {
  if (payload.chart_type === 'histogram') return [{ key: 'count', label: 'Frequency' }];
  if (payload.chart_type === 'mirroredBar') {
    return [
      { key: 'current_weight_mirrored', label: 'Current exposure' },
      { key: 'advisory_weight', label: 'Advisory exposure' },
    ];
  }
  if (payload.chart_type === 'rangeBar') {
    return [{ key: 'range', label: definition?.title || payload.title || 'Range' }];
  }
  return [{ key: payload.y_axis || 'value', label: fieldLabel(payload.y_axis || 'value') }];
}

function applyPremiumFallback(payload, fallbackChart) {
  const next = {
    ...payload,
    chart_type: 'bar',
    component: 'BarChart',
    fallback_used: true,
    status: 'premium_unavailable',
    warnings: [...(payload.warnings || []), premiumUnavailableMessage(fallbackChart)],
  };

  if (fallbackChart === 'standard_min_max_bar') {
    next.bar_mode = 'grouped';
    next.series = [
      { key: payload.rangeStartField, label: 'Minimum', color: DEFAULT_COLORS[0] },
      { key: payload.rangeEndField, label: 'Maximum', color: DEFAULT_COLORS[1] },
    ].filter((series) => series.key);
    next.x_axis = payload.x_axis;
  } else if (fallbackChart === 'contribution_bar') {
    next.bar_mode = 'diverging';
    next.y_axis = 'contribution_value';
    next.series = [{ key: 'contribution_value', label: 'Contribution', color: DEFAULT_COLORS[0] }];
    next.data = payload.data.map((row) => ({
      ...row,
      contribution_value: Number(row?.[payload.rangeEndField]) - Number(row?.[payload.rangeStartField]),
    }));
  } else if (fallbackChart === 'grouped_bar') {
    next.bar_mode = 'grouped';
    next.x_axis = 'ticker';
    next.y_axis = 'weight_percent';
    next.data = payload.data.map((row) => ({
      ...row,
      current_weight_mirrored: -Math.abs(Number(row.current_weight ?? 0)),
    }));
    next.series = [
      { key: 'current_weight_mirrored', label: 'Current exposure', color: DEFAULT_COLORS[0] },
      { key: 'advisory_weight', label: 'Advisory exposure', color: DEFAULT_COLORS[1] },
    ];
  }
  return next;
}

function buildMuiBarSpec(payload) {
  if (payload.chart_type === 'rangeBar' && payload.requires_premium && payload.premium_enabled) {
    return buildRangeBarSpec(payload);
  }
  if (payload.chart_type === 'histogram') {
    return buildHistogramSpec(payload);
  }
  if (payload.chart_type === 'mirroredBar') {
    return buildMirroredBarSpec(payload);
  }
  return buildStandardBarSpec(payload);
}

function buildRangeBarSpec(payload) {
  const categoryKey = inferCategoryKey(payload, payload.rangeStartField);
  const sortedData = sortAndLimitData(payload.data, payload.sort, payload.rangeEndField, payload.top_n);
  const categories = sortedData.map((row) => row[categoryKey]);
  const categoryWidth = categoryAxisWidth(categories);
  const isWaterfall = String(payload.plot_id).includes('waterfall');
  const chartHeight = Math.max(280, Math.min(760, payload.height || 360));
  const series = [{
    type: 'rangeBar',
    label: payload.series?.[0]?.label || payload.title || 'Range',
    datasetKeys: { start: payload.rangeStartField, end: payload.rangeEndField },
    valueFormatter: (value) => formatRangeValue(value, payload.unit),
    highlightScope: { highlight: 'item', fade: 'global' },
    ...(isWaterfall ? { colorGetter: waterfallColorGetter } : { color: payload.series?.[0]?.color || DEFAULT_COLORS[0] }),
  }];

  return {
    valid: sortedData.length > 0 && Boolean(payload.rangeStartField && payload.rangeEndField),
    reason: sortedData.length ? '' : 'No rows available for this range bar chart.',
    payload,
    dataset: sortedData,
    series,
    layout: payload.layout || 'vertical',
    usePremiumRenderer: true,
    componentName: 'BarChartPremium',
    renderer: payload.renderer,
    categories,
    chartHeight,
    margin: { top: 32, right: 28, left: 72, bottom: categories.length > 6 ? 88 : 58 },
    xAxis: [{
      dataKey: categoryKey,
      data: categories,
      scaleType: 'band',
      tickPlacement: 'middle',
      tickLabelPlacement: 'middle',
      label: fieldLabel(payload.x_axis),
      categoryGapRatio: payload.categoryGapRatio ?? 0.32,
      barGapRatio: payload.barGapRatio ?? 0.12,
    }],
    yAxis: [{
      label: fieldLabel(payload.y_axis),
      valueFormatter: getFormatter(payload.unit),
      domainLimit: 'nice',
      width: Math.max(64, Math.min(100, categoryWidth)),
    }],
    grid: payload.grid || { horizontal: true },
    borderRadius: payload.borderRadius ?? 6,
    thresholds: payload.thresholds || [],
    warnings: payload.warnings || [],
    interpretation: payload.interpretation,
    pointCount: sortedData.length,
  };
}

function buildHistogramSpec(payload) {
  const data = payload.data.map((row) => ({
    ...row,
    return_bin: row.return_bin || `${Number(row.bin_start).toFixed(1)}-${Number(row.bin_end).toFixed(1)}`,
  }));
  return buildStandardBarSpec({
    ...payload,
    chart_type: 'bar',
    bar_mode: 'vertical',
    data,
    x_axis: 'return_bin',
    y_axis: 'count',
    categoryGapRatio: payload.categoryGapRatio ?? 0,
    barGapRatio: payload.barGapRatio ?? 0,
    series: [{ key: 'count', label: 'Frequency', color: payload.series?.[0]?.color || DEFAULT_COLORS[0] }],
  });
}

function buildMirroredBarSpec(payload) {
  const data = payload.data.map((row) => ({
    ...row,
    current_weight_mirrored: row.current_weight_mirrored ?? -Math.abs(Number(row.current_weight ?? 0)),
    advisory_weight: Number(row.advisory_weight ?? 0),
  }));
  return buildStandardBarSpec({
    ...payload,
    chart_type: 'bar',
    bar_mode: 'horizontal',
    data,
    x_axis: 'weight_percent',
    y_axis: 'ticker',
    series: [
      { key: 'current_weight_mirrored', label: 'Current exposure', color: DEFAULT_COLORS[0] },
      { key: 'advisory_weight', label: 'Advisory exposure', color: DEFAULT_COLORS[1] },
    ],
  });
}

function buildStandardBarSpec(payload) {
  const mainMetric = getMainMetric(payload);
  const categoryKey = inferCategoryKey(payload, mainMetric);
  const sortedData = sortAndLimitData(payload.data, payload.sort, mainMetric, payload.top_n);
  const layout = payload.layout || chooseLayout({ ...payload, data: sortedData });
  const categories = sortedData.map((row) => row[categoryKey]);
  const categoryWidth = categoryAxisWidth(categories);
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

  const xAxisOverride = payload.xAxis?.[0] || {};
  const yAxisOverride = payload.yAxis?.[0] || {};

  const numericAxis = {
    label: fieldLabel(layout === 'horizontal' ? payload.x_axis : payload.y_axis),
    valueFormatter: (value) => formatValue(Math.abs(Number(value)), payload.unit),
    domainLimit: 'nice',
    ...(layout === 'horizontal' ? xAxisOverride : yAxisOverride),
  };
  const categoryAxis = {
    dataKey: categoryKey,
    data: categories,
    scaleType: 'band',
    tickPlacement: 'middle',
    tickLabelPlacement: 'middle',
    ...(payload.categoryGapRatio != null ? { categoryGapRatio: payload.categoryGapRatio } : { categoryGapRatio: 0.32 }),
    ...(payload.barGapRatio != null ? { barGapRatio: payload.barGapRatio } : { barGapRatio: 0.12 }),
    ...(layout === 'horizontal' ? yAxisOverride : xAxisOverride),
  };

  return {
    valid: sortedData.length > 0 && series.length > 0,
    reason: sortedData.length ? '' : 'No rows available for this bar chart.',
    payload,
    dataset: sortedData,
    series,
    layout,
    usePremiumRenderer: false,
    componentName: 'BarChart',
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

function waterfallColorGetter(data) {
  const value = data?.value;
  if (!Array.isArray(value) || value[0] === 0) return DEFAULT_COLORS[0];
  return value[1] - value[0] >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
}

function categoryAxisWidth(categories) {
  return Math.max(78, Math.min(220, categories.reduce((max, label) => Math.max(max, String(label ?? '').length), 0) * 7 + 28));
}

function fieldLabel(field) {
  const key = String(field || '').trim();
  return FIELD_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) || 'Value';
}

function inferCategoryKey(payload, mainMetric) {
  const candidates = [payload.x_axis, payload.y_axis, 'ticker', 'sector', 'strategy', 'portfolio', 'component', 'return_bin', 'category', 'label'];
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

function formatRangeValue(value, unit) {
  if (!Array.isArray(value)) return '';
  return `${formatValue(value[0], unit)} - ${formatValue(value[1], unit)}`;
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
  return { valid: false, reason, dataset: [], series: [], xAxis: [], yAxis: [], chartHeight: 260, margin: {}, usePremiumRenderer: false };
}






