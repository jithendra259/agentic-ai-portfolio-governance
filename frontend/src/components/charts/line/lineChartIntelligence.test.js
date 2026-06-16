import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { CHART_TIER } from '../chartTierConfig.js';
import { getLineChartDefinition } from './lineChartRegistry.js';
import { validateLineChartPayload } from './lineChartValidator.js';
import { prepareSeries } from './lineChartUtils.js';

function baseLinePayload(overrides = {}) {
  return {
    plot_type: 'line',
    plot_id: 'historical_adjusted_close',
    chart_type: 'line',
    chart_tier: CHART_TIER.FREE,
    component: 'LineChart',
    title: 'Historical Adjusted Close',
    tickers_used: ['AAPL'],
    ticker_count: 1,
    x_axis: 'date',
    y_axis: 'adjusted_close',
    unit: 'price',
    connect_nulls: false,
    curve: 'linear',
    optimizer_called: false,
    advisory_allocation_generated: false,
    required_fields: ['date', 'adjusted_close'],
    data: [
      { date: '2024-01-02', ticker: 'AAPL', adjusted_close: 100 },
      { date: '2024-01-03', ticker: 'AAPL', adjusted_close: 101 },
    ],
    series: [
      {
        name: 'AAPL',
        label: 'AAPL',
        connectNulls: false,
        data: [
          { x: '2024-01-02', y: 100 },
          { x: '2024-01-03', y: 101 },
        ],
      },
    ],
    ...overrides,
  };
}

describe('line chart registry', () => {
  it('registers instability index as optional premium with a free fallback', () => {
    const definition = getLineChartDefinition('instability_index_over_time');
    assert.equal(definition.chart_tier, CHART_TIER.PREMIUM);
    assert.equal(definition.requires_premium, true);
    assert.equal(definition.fallback_chart, 'basic_instability_line');
  });

  it('registers normalized comparison with the normalization fallback method', () => {
    const definition = getLineChartDefinition('normalized_price_comparison');
    assert.equal(definition.fallback_computation, 'normalize_adjusted_close_to_100');
    assert.deepEqual(definition.required_fields, ['date', 'ticker', 'normalized_price']);
  });
});

describe('line chart validation', () => {
  it('accepts a single ticker adjusted close line with no optimizer', () => {
    const result = validateLineChartPayload(baseLinePayload());
    assert.equal(result.valid, true);
  });

  it('blocks unsorted line dates', () => {
    const result = validateLineChartPayload(baseLinePayload({
      data: [
        { date: '2024-01-03', ticker: 'AAPL', adjusted_close: 101 },
        { date: '2024-01-02', ticker: 'AAPL', adjusted_close: 100 },
      ],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /sorted ascending/);
  });

  it('blocks all-null y values', () => {
    const result = validateLineChartPayload(baseLinePayload({
      data: [
        { date: '2024-01-02', ticker: 'AAPL', adjusted_close: null },
        { date: '2024-01-03', ticker: 'AAPL', adjusted_close: null },
      ],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /all null/);
  });

  it('blocks raw multi-ticker adjusted close lines', () => {
    const result = validateLineChartPayload(baseLinePayload({
      tickers_used: ['AAPL', 'MSFT'],
      ticker_count: 2,
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /normalized_price_comparison/);
  });

  it('warns and allows premium line fallback when premium is disabled', () => {
    const result = validateLineChartPayload(baseLinePayload({
      plot_id: 'instability_index_over_time',
      chart_tier: CHART_TIER.PREMIUM,
      component: 'ChartsDataProviderPro',
      requires_premium: true,
      fallback_chart: 'basic_instability_line',
      y_axis: 'instability_index',
      required_fields: ['date', 'instability_index'],
      data: [
        { date: '2024-01-02', instability_index: 0.2, calm_threshold: 0.5, crisis_threshold: 0.85 },
        { date: '2024-01-03', instability_index: 0.4, calm_threshold: 0.5, crisis_threshold: 0.85 },
      ],
      series: [{ name: 'instability_index', data: [{ x: '2024-01-02', y: 0.2 }] }],
    }), { premiumEnabled: false });
    assert.equal(result.valid, true);
    assert.equal(result.premium_unavailable, true);
    assert.match(result.warnings.join('\n'), /fallback/);
  });

  it('blocks optimizer leakage on line-only plots', () => {
    const result = validateLineChartPayload(baseLinePayload({ optimizer_called: true }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /optimizer/);
  });
});

describe('line chart rendering adapter', () => {
  it('converts backend end-marker shorthand into a MUI showMark callback', () => {
    const series = prepareSeries(baseLinePayload({
      series: [
        {
          name: 'AAPL',
          data: [
            { x: '2024-01-02', y: 100 },
            { x: '2024-01-03', y: 101 },
          ],
          showMark: 'end',
        },
      ],
    }), false);

    assert.equal(typeof series[0].showMark, 'function');
    assert.equal(series[0].showMark({ index: 0 }), false);
    assert.equal(series[0].showMark({ index: 1 }), true);
  });
});

