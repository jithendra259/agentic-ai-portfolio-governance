import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildLineSeries,
  getActiveRegime,
  getDates,
  getDateRangeForPreset,
  getSeriesDataArray,
  getUniverseTickers,
} from './analyticsDashboardModel.js';

test('resolves dashboard universe tickers with U1 fallback', () => {
  assert.deepEqual(getUniverseTickers('U2'), ['BAC', 'GS', 'WFC', 'BLK', 'AXP']);
  assert.deepEqual(getUniverseTickers('UNKNOWN'), ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'JPM']);
});

test('resolves supported dashboard date presets', () => {
  assert.deepEqual(getDateRangeForPreset('2023'), {
    startDate: '2023-01-01',
    endDate: '2023-12-31',
  });
  assert.deepEqual(getDateRangeForPreset('not-set'), {
    startDate: '2024-01-01',
    endDate: '2024-12-31',
  });
});

test('maps chart rows into dates and numeric series arrays', () => {
  const rows = [
    { date: '2024-01-01', AAPL: 10 },
    { date: '2024-01-02' },
  ];

  assert.deepEqual(getDates(rows).map((date) => date.toISOString().slice(0, 10)), [
    '2024-01-01',
    '2024-01-02',
  ]);
  assert.deepEqual(getSeriesDataArray(rows, 'AAPL'), [10, 0]);
  assert.deepEqual(buildLineSeries(rows, ['AAPL']), [
    { data: [10, 0], label: 'AAPL', showMark: false },
  ]);
});

test('uses the latest regime from timeline with calm fallback', () => {
  assert.equal(getActiveRegime({ regime_timeline: [{ regime: 'Calm' }, { regime: 'Crisis' }] }), 'Crisis');
  assert.equal(getActiveRegime({ regime_timeline: [] }), 'Calm');
  assert.equal(getActiveRegime(null), 'Calm');
});
