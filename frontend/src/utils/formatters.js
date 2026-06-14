export function toFiniteNumber(value, fallback = null) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function formatCurrency(value, currency = 'USD') {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(numeric);
}

export function formatCompactCurrency(value, currency = 'USD') {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(numeric);
}

export function formatPercent(value, options = {}) {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '-';
  const normalized = options.alreadyPercent ? numeric / 100 : numeric;
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: options.minimumFractionDigits ?? 2,
    maximumFractionDigits: options.maximumFractionDigits ?? 2,
  }).format(normalized);
}

export function formatDecimal(value, digits = 2) {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '-';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(numeric);
}

export function formatFinancialValue(value, format) {
  if (format === 'currency') return formatCurrency(value);
  if (format === 'compactCurrency') return formatCompactCurrency(value);
  if (format === 'percent' || format === '%') return formatPercent(value);
  if (format === 'percentPoints') return formatPercent(value, { alreadyPercent: true });
  if (format === 'decimal') return formatDecimal(value);
  if (format === 'beta') {
    const numeric = toFiniteNumber(value);
    return numeric == null ? '-' : `${formatDecimal(numeric)} beta`;
  }
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '-';
  if (Math.abs(numeric) >= 1000000) return `${formatDecimal(numeric / 1000000, 1)}M`;
  if (Math.abs(numeric) >= 1000) return `${formatDecimal(numeric / 1000, 1)}K`;
  return formatDecimal(numeric);
}
