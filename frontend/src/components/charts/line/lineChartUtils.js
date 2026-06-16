import { formatFinancialValue } from '../../../utils/formatters.js';
import { dateParseMode, parseDateValue } from '../../../utils/plotDataParser.js';

export const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
export const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };
export const GRID_STYLE = { stroke: '#374151', strokeDasharray: '4 4' };

export function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function getValueFormatter(format) {
  if (format === 'percent' || format === '%') return (v) => formatFinancialValue(v, 'percentPoints');
  if (format === 'decimal') return (v) => formatFinancialValue(v, 'decimal');
  if (format === 'beta') return (v) => formatFinancialValue(v, 'beta');
  if (format === 'k') return (v) => (v == null ? '' : `${(v / 1000).toFixed(1)}k`);
  if (format === 'currency') return (v) => formatFinancialValue(v, 'currency');
  if (format === 'compactCurrency') return (v) => formatFinancialValue(v, 'compactCurrency');
  return (v) => {
    const formatted = formatFinancialValue(v);
    return formatted === '-' ? '' : formatted;
  };
}

export function prepareDataset(seriesSpec, spec = {}) {
  if (!seriesSpec) return [];
  const mode = dateParseMode(spec);
  const dateSet = new Set();
  seriesSpec.forEach((s) => s.data?.forEach((pt) => {
    const date = parseDateValue(pt.x ?? pt.date, { mode });
    if (date) dateSet.add(date.toISOString());
  }));
  const sortedDates = Array.from(dateSet).sort();
  const byDate = {};
  sortedDates.forEach((d) => { byDate[d] = { date: new Date(d) }; });
  seriesSpec.forEach((s) => {
    s.data?.forEach((pt) => {
      const date = parseDateValue(pt.x ?? pt.date, { mode });
      const key = date?.toISOString();
      if (key && byDate[key]) byDate[key][s.name] = toFiniteNumber(pt.y ?? pt.value, null);
    });
  });
  return sortedDates.map((d) => byDate[d]);
}

export function prepareSeries(spec, inferredArea) {
  if (!spec?.series) return [];
  return spec.series.map((s, i) => {
    const isAreaChart = spec.chart_type === 'line_area' || spec.chart_type === 'stacked_area' || inferredArea;
    const showMark = resolveShowMark(s.showMark, s.data);
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
    entry.showMark = showMark;
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

function resolveShowMark(showMark, points = []) {
  if (showMark === 'end') {
    const lastIndex = Array.isArray(points) ? points.length - 1 : -1;
    return ({ index }) => index === lastIndex;
  }
  if (typeof showMark === 'boolean' || typeof showMark === 'function') return showMark;
  return false;
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
