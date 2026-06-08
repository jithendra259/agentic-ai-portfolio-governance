export function getResponsiveChartHeight(spec, fallback = 360) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}
