export function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function getResponsiveChartHeight(spec, fallback = 64) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(32, Math.min(requested, 180)) : fallback;
}
