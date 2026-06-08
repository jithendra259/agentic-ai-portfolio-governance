export const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
export const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };
export const GRID_STYLE = { stroke: '#374151', strokeDasharray: '4 4' };

export function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function getValueFormatter(format) {
  if (format === 'percent' || format === '%') return (v) => (v == null ? '' : `${v.toFixed(1)}%`);
  if (format === 'decimal') return (v) => (v == null ? '' : Number(v).toFixed(2));
  if (format === 'beta') return (v) => (v == null ? '' : `${Number(v).toFixed(2)} beta`);
  if (format === 'k') return (v) => (v == null ? '' : `${(v / 1000).toFixed(1)}k`);
  if (format === 'currency') return (v) => (v == null ? '' : `$${v.toLocaleString()}`);
  return (v) => {
    if (v == null) return '';
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return Number(v).toFixed(2);
  };
}

export function prepareDataset(seriesSpec) {
  if (!seriesSpec) return [];
  const dateSet = new Set();
  seriesSpec.forEach((s) => s.data?.forEach((pt) => dateSet.add(pt.x)));
  const sortedDates = Array.from(dateSet).sort();
  const byDate = {};
  sortedDates.forEach((d) => { byDate[d] = { date: new Date(d) }; });
  seriesSpec.forEach((s) => {
    s.data?.forEach((pt) => {
      if (byDate[pt.x]) byDate[pt.x][s.name] = pt.y;
    });
  });
  return sortedDates.map((d) => byDate[d]);
}

export function prepareSeries(spec, inferredArea) {
  if (!spec?.series) return [];
  return spec.series.map((s, i) => {
    const isAreaChart = spec.chart_type === 'line_area' || spec.chart_type === 'stacked_area' || inferredArea;
    const entry = {
      type: 'line',
      dataKey: s.name,
      label: s.label || s.name,
      color: s.color || PALETTE[i % PALETTE.length],
      valueFormatter: getValueFormatter(s.value_format || spec.y_format),
    };
    if (s.yAxisId) entry.yAxisId = s.yAxisId;
    entry.area = s.area ?? isAreaChart;
    if (s.baseline != null) entry.baseline = s.baseline;
    if (s.stack) entry.stack = s.stack;
    if (s.stackOffset) entry.stackOffset = s.stackOffset;
    entry.showMark = s.showMark ?? false;
    if (s.shape) entry.shape = s.shape;
    entry.connectNulls = s.connectNulls ?? spec.connect_nulls ?? spec.connectNulls ?? false;
    if (s.highlightScope) entry.highlightScope = s.highlightScope;
    else if (spec.highlightScope) entry.highlightScope = spec.highlightScope;
    if (s.disableHighlight != null) entry.disableHighlight = s.disableHighlight;
    if (s.curve) entry.curve = s.curve;
    else if (spec.curve) entry.curve = spec.curve;
    return entry;
  });
}

export function prepareYAxis(spec) {
  if (spec.yAxis && Array.isArray(spec.yAxis)) {
    return spec.yAxis.map((axis) => ({
      id: axis.id,
      label: axis.label || '',
      position: axis.position || 'left',
      tickLabelStyle: AXIS_STYLE,
      valueFormatter: getValueFormatter(axis.value_format),
      width: axis.width || (axis.position === 'right' ? 50 : 55),
      domainLimit: axis.domainLimit || 'nice',
      ...(axis.colorMap ? { colorMap: axis.colorMap } : {}),
    }));
  }
  return [{
    id: 'default-y-axis',
    tickLabelStyle: AXIS_STYLE,
    label: spec.y_label || '',
    valueFormatter: getValueFormatter(spec.y_format),
    domainLimit: 'nice',
  }];
}

export function prepareMargins(spec) {
  const hasRightAxis = spec.yAxis?.some((axis) => axis.position === 'right');
  return { top: 60, right: hasRightAxis ? 60 : 24, left: 60, bottom: 60 };
}
