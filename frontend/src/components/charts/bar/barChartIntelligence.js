export function inferBarMode(payload) {
  const explicit = payload?.bar_mode;
  if (explicit && explicit !== 'vertical') return explicit;

  const title = `${payload?.title || ''} ${payload?.plot_id || ''}`.toLowerCase();
  const seriesCount = Array.isArray(payload?.series) ? payload.series.length : 0;
  const data = Array.isArray(payload?.data) ? payload.data : [];
  const categoryCount = data.length || getLegacyCategoryCount(payload);

  if (title.includes('current vs advisory') || title.includes('allocation vs risk')) return 'grouped';
  if (title.includes('component contribution') || payload?.series?.some((s) => s.stack)) return 'stacked';
  if (title.includes('change') || hasPositiveAndNegativeValues(payload)) return 'diverging';
  if (seriesCount > 1) return 'grouped';
  if (categoryCount > 8) return 'horizontal';
  return explicit || 'vertical';
}

export function chooseLayout(payload) {
  const mode = inferBarMode(payload);
  const categoryCount = Array.isArray(payload?.data) ? payload.data.length : getLegacyCategoryCount(payload);
  if (mode === 'horizontal' || categoryCount > 8) return 'horizontal';
  return 'vertical';
}

export function chooseRenderer(payload) {
  const pointCount = getPointCount(payload);
  const requested = payload?.renderer;
  if (requested) return requested;
  if (pointCount >= 5000 && payload?.premium_enabled && payload?.component === 'BarChartPremium') return 'webgl';
  if (pointCount >= 500) return 'svg-batch';
  return 'svg-single';
}

export function shouldShowLabels(payload) {
  const pointCount = getPointCount(payload);
  if (payload?.show_labels != null) return Boolean(payload.show_labels);
  return pointCount <= 60;
}

export function getMainMetric(payload) {
  const firstSeries = Array.isArray(payload?.series) ? payload.series[0] : null;
  if (firstSeries?.key || firstSeries?.name) return firstSeries.key || firstSeries.name;
  if (payload?.y_axis && payload?.bar_mode !== 'horizontal') return payload.y_axis;
  if (payload?.x_axis && payload?.bar_mode === 'horizontal') return payload.x_axis;
  return 'value';
}

function getLegacyCategoryCount(payload) {
  const categories = new Set();
  payload?.series?.forEach((series) => series?.data?.forEach((point) => categories.add(point.x)));
  return categories.size;
}

function getPointCount(payload) {
  if (Array.isArray(payload?.data)) return payload.data.length * Math.max(1, payload?.series?.length || 1);
  return payload?.series?.reduce((sum, series) => sum + (series?.data?.length || 0), 0) || 0;
}

function hasPositiveAndNegativeValues(payload) {
  const values = [];
  if (Array.isArray(payload?.data) && Array.isArray(payload?.series)) {
    payload.data.forEach((row) => {
      payload.series.forEach((series) => values.push(Number(row?.[series.key])));
    });
  } else {
    payload?.series?.forEach((series) => series?.data?.forEach((point) => values.push(Number(point.y))));
  }
  return values.some((value) => value > 0) && values.some((value) => value < 0);
}


