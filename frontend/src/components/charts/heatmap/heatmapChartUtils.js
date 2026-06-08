export const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };

export function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function getResponsiveChartHeight(spec, fallback = 420) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}
