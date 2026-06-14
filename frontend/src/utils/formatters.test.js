import assert from 'node:assert/strict';
import test from 'node:test';

import { formatCurrency, formatPercent, formatFinancialValue } from './formatters.js';

test('formatCurrency rounds noisy floats for display', () => {
  assert.equal(formatCurrency(145.39847500000002), '$145.40');
});

test('formatPercent supports decimal and percent-point inputs', () => {
  assert.equal(formatPercent(0.1234), '12.34%');
  assert.equal(formatFinancialValue(12.34, 'percentPoints'), '12.34%');
});
