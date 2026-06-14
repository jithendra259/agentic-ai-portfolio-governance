import { toFiniteNumber } from './formatters.js';

const DATE_ONLY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export function parseSafeLocalDate(value) {
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
  if (typeof value === 'string' && DATE_ONLY_PATTERN.test(value)) {
    const [year, month, day] = value.split('-').map(Number);
    return new Date(year, month - 1, day, 12, 0, 0, 0);
  }
  return parseDateValue(value, { mode: 'timestamp' });
}

export function parseDateValue(value, options = {}) {
  const mode = options.mode || 'auto';
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
  if (typeof value === 'number' && Number.isFinite(value)) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  if (typeof value === 'string' && value.trim()) {
    if ((mode === 'date_only' || mode === 'auto') && DATE_ONLY_PATTERN.test(value)) {
      const [year, month, day] = value.split('-').map(Number);
      return new Date(year, month - 1, day, 12, 0, 0, 0);
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  return null;
}

export function isTimeScaleSpec(plotSpec) {
  const xType = plotSpec?.x_type || plotSpec?.xAxisType;
  return xType === 'time' || xType === 'utc' || xType === 'date_only' || plotSpec?.x_scale === 'time';
}

export function dateParseMode(plotSpec) {
  const xType = plotSpec?.x_type || plotSpec?.xAxisType;
  return xType === 'date_only' ? 'date_only' : 'timestamp';
}

export function dateScaleType(plotSpec) {
  const xType = plotSpec?.x_type || plotSpec?.xAxisType;
  if (xType === 'utc') return 'utc';
  return 'time';
}

export function formatChartDate(date, plotSpec) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const timeZone = (plotSpec?.x_type || plotSpec?.xAxisType) === 'utc' ? 'UTC' : undefined;
  return new Intl.DateTimeFormat('en-US', {
    ...(timeZone ? { timeZone } : {}),
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(date);
}

export function parsePlotDataForMUI(plotSpec) {
  if (!plotSpec || !Array.isArray(plotSpec.series)) return plotSpec;
  const isTimeScale = isTimeScaleSpec(plotSpec);
  const mode = dateParseMode(plotSpec);

  return {
    ...plotSpec,
    series: plotSpec.series.map((series, seriesIndex) => ({
      ...series,
      data: (series.data || [])
        .map((point, pointIndex) => {
          const rawX = point.x ?? point.date;
          const x = isTimeScale ? parseDateValue(rawX, { mode }) : rawX;
          if (isTimeScale && !x) return null;
          const y = toFiniteNumber(point.y ?? point.value);
          if (y == null) return null;
          return {
            ...point,
            x,
            y,
            id: point.id ?? `${series.name || series.label || seriesIndex}-${x instanceof Date ? x.getTime() : x}-${pointIndex}`,
          };
        })
        .filter(Boolean),
    })),
  };
}
