export const METRIC_CONFIG = {
  sharpeRatio: { min: -1, max: 3, label: 'Sharpe Ratio', higherIsBetter: true, rawFormat: 'number' },
  sharpe: { min: -1, max: 3, label: 'Sharpe Ratio', higherIsBetter: true, rawFormat: 'number' },
  sortino: { min: -1, max: 4, label: 'Sortino Ratio', higherIsBetter: true, rawFormat: 'number' },
  annualReturn: { min: -30, max: 50, label: 'Annual Return', higherIsBetter: true, rawFormat: 'percentPoints' },
  totalReturn: { min: -60, max: 120, label: 'Total Return', higherIsBetter: true, rawFormat: 'percentPoints' },
  volatility: { min: 0, max: 50, label: 'Volatility', higherIsBetter: false, rawFormat: 'percentPoints' },
  maxDrawdown: { min: -60, max: 0, label: 'Max Drawdown', higherIsBetter: false, rawFormat: 'percentPoints' },
  cvar95: { min: -40, max: 0, label: 'CVaR 95%', higherIsBetter: false, rawFormat: 'percentPoints' },
  alpha: { min: -15, max: 20, label: 'Alpha', higherIsBetter: true, rawFormat: 'percentPoints' },
  beta: { min: 0, max: 2, label: 'Beta', higherIsBetter: false, rawFormat: 'number' },
};

export function getMetricConfig(metric) {
  if (!metric) return null;
  if (typeof metric === 'string') {
    return METRIC_CONFIG[metric] || { min: 0, max: 100, label: metric, higherIsBetter: true, rawFormat: 'number' };
  }
  const key = metric.key || metric.dataKey || metric.name || metric.id;
  const base = METRIC_CONFIG[key] || {};
  return {
    min: metric.min ?? base.min ?? 0,
    max: metric.max ?? base.max ?? 100,
    label: metric.label ?? base.label ?? key,
    higherIsBetter: metric.higherIsBetter ?? base.higherIsBetter ?? true,
    rawFormat: metric.rawFormat ?? metric.value_format ?? base.rawFormat ?? 'number',
  };
}
