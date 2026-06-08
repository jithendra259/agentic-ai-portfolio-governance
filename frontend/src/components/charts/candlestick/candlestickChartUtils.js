export function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function getResponsiveChartHeight(spec, fallback = 320) {
  const requested = Number(spec?.height);
  if (!Number.isFinite(requested)) return fallback;
  return Math.max(180, Math.min(requested, 720));
}

export function formatDateLabel(date, labelMode) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const monthDay = { month: 'short', day: 'numeric' };
  const monthYear = { month: 'short', year: 'numeric' };
  const fullDate = { month: 'short', day: 'numeric', year: 'numeric' };

  if (labelMode === 'year') return date.getFullYear().toString();
  if (labelMode === 'quarter') {
    const quarter = Math.floor(date.getMonth() / 3) + 1;
    return `Q${quarter} ${date.getFullYear()}`;
  }
  if (labelMode === 'month') return date.toLocaleDateString('en-US', monthYear);
  if (labelMode === 'day') return date.toLocaleDateString('en-US', monthDay);
  return date.toLocaleDateString('en-US', fullDate);
}

export function getLabelModeFromSpan(spanDays, spanYears) {
  if (spanYears >= 10) return 'year';
  if (spanYears >= 3) return 'quarter';
  if (spanYears >= 1) return 'month';
  if (spanDays > 90) return 'month';
  if (spanDays > 45) return 'month';
  if (spanDays > 14) return 'week';
  return 'day';
}

export function formatVolume(val) {
  if (val == null) return '';
  if (val >= 1000000000) return `${(val / 1000000000).toFixed(1)}B`;
  if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
  if (val >= 1000) return `${(val / 1000).toFixed(1)}k`;
  return val.toString();
}

export function formatAsDollar(value) {
  if (value == null) return '';
  return `$${value.toLocaleString('en-US', { maximumFractionDigits: value >= 100 ? 0 : 2 })}`;
}

export function layoutCandlestick({ pts, containerWidth, height }) {
  const data = pts
    .map((entry, index) => ({
      index,
      date: entry.date,
      dateObj: entry.date ? new Date(entry.date) : null,
      open: toFiniteNumber(entry.open),
      high: toFiniteNumber(entry.high),
      low: toFiniteNumber(entry.low),
      close: toFiniteNumber(entry.close),
      volume: Math.max(0, toFiniteNumber(entry.volume)),
    }))
    .sort((a, b) => {
      const aTime = a.dateObj instanceof Date && !Number.isNaN(a.dateObj.getTime()) ? a.dateObj.getTime() : 0;
      const bTime = b.dateObj instanceof Date && !Number.isNaN(b.dateObj.getTime()) ? b.dateObj.getTime() : 0;
      return aTime - bTime;
    })
    .map((entry, index) => ({ ...entry, index }));

  const margin = { top: 22, right: 64, bottom: 42, left: 34 };
  const innerWidth = Math.max(260, containerWidth - margin.left - margin.right);
  const innerHeight = Math.max(200, height - margin.top - margin.bottom);
  const volumeHeight = data.some((entry) => entry.volume > 0) ? Math.max(46, innerHeight * 0.22) : 0;
  const volumeGap = volumeHeight ? 12 : 0;
  const priceHeight = innerHeight - volumeHeight - volumeGap;
  const plotBottom = margin.top + priceHeight;
  const minLow = Math.min(...data.map((entry) => entry.low));
  const maxHigh = Math.max(...data.map((entry) => entry.high));
  const pricePadding = Math.max((maxHigh - minLow) * 0.1, 1);
  const minPrice = minLow - pricePadding;
  const maxPrice = maxHigh + pricePadding;
  const priceRange = Math.max(maxPrice - minPrice, 1);
  const maxVolume = Math.max(1, ...data.map((entry) => entry.volume));
  const step = innerWidth / Math.max(1, data.length);
  const candleWidth = Math.max(5, Math.min(16, step * 0.58));
  const denseMode = data.length > 60;
  
  const xFor = (index) => margin.left + step * (index + 0.5);
  const priceY = (value) => margin.top + ((maxPrice - value) / priceRange) * priceHeight;
  const volumeY = (value) => plotBottom + volumeGap + volumeHeight - (value / maxVolume) * volumeHeight;
  
  const priceTicks = Array.from({ length: 5 }, (_, index) => {
    const value = minPrice + (priceRange * index) / 4;
    return { value, y: priceY(value) };
  }).reverse();
  
  const windowSize = 20;
  const movingAverage = data.map((_, index) => {
    if (index < windowSize - 1) return null;
    const window = data.slice(index - windowSize + 1, index + 1);
    return window.reduce((sum, entry) => sum + entry.close, 0) / window.length;
  });
  
  const movingAveragePath = movingAverage
    .map((value, index) => value == null ? null : `${index === windowSize - 1 ? 'M' : 'L'} ${xFor(index)} ${priceY(value)}`)
    .filter(Boolean)
    .join(' ');
    
  const firstDate = data[0]?.dateObj;
  const lastDate = data[data.length - 1]?.dateObj;
  const spanMs = firstDate && lastDate && !Number.isNaN(firstDate.getTime()) && !Number.isNaN(lastDate.getTime())
    ? Math.max(0, lastDate.getTime() - firstDate.getTime())
    : 0;
  const spanDays = spanMs / (1000 * 60 * 60 * 24);
  const spanYears = spanDays / 365.25;
  const labelMode = getLabelModeFromSpan(spanDays, spanYears);
  
  const tickEvery = labelMode === 'year'
    ? Math.max(1, Math.ceil(data.length / Math.min(12, Math.max(2, Math.round(spanYears) || 1))))
    : labelMode === 'quarter'
      ? Math.max(1, Math.ceil(data.length / Math.min(16, Math.max(4, Math.round(spanYears * 4) || 4))))
      : labelMode === 'month'
        ? Math.max(1, Math.ceil(data.length / Math.min(12, Math.max(4, Math.round(spanYears * 12) || 6))))
        : Math.max(1, Math.ceil(data.length / 7));
        
  const dateTicks = data
    .filter((_, index) => index === 0 || index === data.length - 1 || index % tickEvery === 0)
    .map((entry) => ({
      x: xFor(entry.index),
      label: entry.dateObj instanceof Date && !Number.isNaN(entry.dateObj.getTime())
        ? formatDateLabel(entry.dateObj, labelMode)
        : entry.date,
    }));

  return { 
    width: containerWidth, 
    height, 
    data, 
    margin, 
    innerWidth, 
    priceHeight, 
    volumeHeight, 
    volumeGap, 
    plotBottom, 
    candleWidth, 
    xFor, 
    priceY, 
    volumeY, 
    priceTicks, 
    dateTicks, 
    movingAverage, 
    movingAveragePath, 
    hasVolume: volumeHeight > 0, 
    denseMode, 
    labelMode 
  };
}
