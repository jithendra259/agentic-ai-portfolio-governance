export const DASHBOARD_UNIVERSES = {
  U1: ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'JPM'],
  U2: ['BAC', 'GS', 'WFC', 'BLK', 'AXP'],
  U3: ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV'],
  U4: ['TSLA', 'MCD', 'NKE', 'SBUX', 'DAL'],
  U5: ['GE', 'HON', 'CAT', 'BA', 'LMT'],
};

export const DASHBOARD_TABS = [
  'Data EDA',
  'Correlation & Covariance',
  'Instability Monitor',
  'Advisory Diversification',
  'Diversification Diagnostics',
  'Risk Governance',
  'Contagion Graph Analysis',
  'Agent Governance & HITL',
  'Evaluation & Backtesting',
];

export function getUniverseTickers(universe) {
  return DASHBOARD_UNIVERSES[universe] || DASHBOARD_UNIVERSES.U1;
}

export function getDateRangeForPreset(datePreset) {
  if (datePreset === '2023') {
    return { startDate: '2023-01-01', endDate: '2023-12-31' };
  }
  return { startDate: '2024-01-01', endDate: '2024-12-31' };
}

export function getDates(seriesData) {
  if (!Array.isArray(seriesData) || seriesData.length === 0) return [];
  return seriesData.map((item) => new Date(item.date));
}

export function getSeriesDataArray(seriesData, key) {
  if (!Array.isArray(seriesData) || seriesData.length === 0) return [];
  return seriesData.map((item) => (item[key] !== undefined ? item[key] : 0.0));
}

export function buildLineSeries(seriesData, keys) {
  if (!Array.isArray(seriesData) || seriesData.length === 0) return [];
  return keys.map((key) => ({
    data: seriesData.map((item) => (item[key] !== undefined ? item[key] : 0.0)),
    label: key,
    showMark: false,
  }));
}

export function getActiveRegime(instabilityData) {
  const timeline = instabilityData?.regime_timeline;
  if (Array.isArray(timeline) && timeline.length > 0) {
    return timeline[timeline.length - 1].regime;
  }
  return 'Calm';
}
