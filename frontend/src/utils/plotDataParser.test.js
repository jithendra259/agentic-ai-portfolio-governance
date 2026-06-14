import assert from 'node:assert/strict';
import test from 'node:test';

import { formatChartDate, parseDateValue, parsePlotDataForMUI, parseSafeLocalDate } from './plotDataParser.js';

test('parseDateValue returns null for invalid dates', () => {
  assert.equal(parseDateValue('not-a-date'), null);
});

test('parsePlotDataForMUI converts ISO x values to Dates for time specs', () => {
  const parsed = parsePlotDataForMUI({
    x_type: 'time',
    series: [{ name: 'AAPL', data: [{ x: '2024-01-02', y: '184.89' }, { x: 'bad', y: 1 }] }],
  });

  assert.equal(parsed.series[0].data.length, 1);
  assert.ok(parsed.series[0].data[0].x instanceof Date);
  assert.equal(parsed.series[0].data[0].y, 184.89);
});

test('parseSafeLocalDate treats date-only strings as local calendar dates', () => {
  const date = parseSafeLocalDate('2023-10-25');

  assert.equal(date.getFullYear(), 2023);
  assert.equal(date.getMonth(), 9);
  assert.equal(date.getDate(), 25);
  assert.equal(date.getHours(), 12);
});

test('formatChartDate uses UTC formatting only for utc specs', () => {
  const date = new Date(Date.UTC(2023, 9, 25, 0, 0, 0));

  assert.equal(formatChartDate(date, { x_type: 'utc' }), 'Oct 25, 2023');
});
