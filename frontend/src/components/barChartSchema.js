export const BAR_MODES = new Set(['vertical', 'horizontal', 'grouped', 'stacked', 'diverging']);
export const BAR_UNITS = new Set(['percent', 'decimal', 'currency', 'count', 'ratio', 'none']);
export const BAR_SORTS = new Set(['none', 'ascending', 'descending']);

export const REQUIRED_BAR_FIELDS = [
  'plot_id',
  'chart_type',
  'bar_mode',
  'title',
  'x_axis',
  'y_axis',
  'unit',
  'series',
  'data',
];

export function isBarPayload(value) {
  return Boolean(value && typeof value === 'object' && (value.chart_type === 'bar' || value.plot_type === 'bar'));
}

export function normalizePlotId(plotId) {
  return String(plotId || '')
    .trim()
    .toLowerCase()
    .replace(/^plot_\d+_/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}
