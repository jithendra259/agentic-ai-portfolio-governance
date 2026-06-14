import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeMetricValue, normalizeRadarSeries } from './radarNormalizer.js';

test('normalizeMetricValue inverts lower-is-better risk metrics', () => {
  const lowRisk = normalizeMetricValue(10, { min: 0, max: 50, higherIsBetter: false });
  const highRisk = normalizeMetricValue(30, { min: 0, max: 50, higherIsBetter: false });

  assert.ok(lowRisk > highRisk);
  assert.equal(lowRisk, 80);
  assert.equal(highRisk, 40);
});

test('normalizeRadarSeries keeps raw data beside normalized geometry', () => {
  const [series] = normalizeRadarSeries(
    [{ label: 'Portfolio A', data: [1.2, 10] }],
    ['sharpeRatio', 'volatility'],
  );

  assert.deepEqual(series.rawData, [1.2, 10]);
  assert.equal(series.data[1], 80);
});
