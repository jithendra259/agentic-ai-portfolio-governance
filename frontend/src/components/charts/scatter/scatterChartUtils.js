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
