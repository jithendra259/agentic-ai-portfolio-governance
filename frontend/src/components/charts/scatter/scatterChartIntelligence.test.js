import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { CHART_TIER } from '../chartTierConfig.js';
import { getScatterChartDefinition } from './scatterChartRegistry.js';
import { validateScatterChartPayload } from './scatterChartValidator.js';

function baseScatterPayload(overrides = {}) {
  return {
    plot_type: 'scatter',
    plot_id: 'risk_return_scatter',
    chart_type: 'scatter',
    chart_tier: CHART_TIER.FREE,
    component: 'ScatterChart',
    title: 'Risk-Return Scatter by Ticker',
    x_axis: 'annualized_volatility_percent',
    y_axis: 'annualized_return_percent',
    x_unit: '%',
    y_unit: '%',
    point_id: 'ticker',
    color_axis: 'sector',
    point_count: 2,
    required_fields: ['ticker', 'annualized_volatility_percent', 'annualized_return_percent'],
    optimizer_called: false,
    advisory_allocation_generated: false,
    data: [
      { ticker: 'AAPL', annualized_volatility_percent: 18.4, annualized_return_percent: 12.6, sector: 'Technology' },
      { ticker: 'MSFT', annualized_volatility_percent: 15.1, annualized_return_percent: 9.2, sector: 'Technology' },
    ],
    series: [
      {
        name: 'Technology',
        data: [
          { id: 'AAPL', x: 18.4, y: 12.6 },
          { id: 'MSFT', x: 15.1, y: 9.2 },
        ],
      },
    ],
    ...overrides,
  };
}

describe('scatter chart registry', () => {
  it('registers portfolio relationship plot families', () => {
    assert.equal(getScatterChartDefinition('risk_return_scatter').x_axis, 'annualized_volatility_percent');
    assert.equal(getScatterChartDefinition('bubble_risk_return').chart_type, 'bubble_scatter');
    assert.equal(getScatterChartDefinition('scatter_with_regression_line').chart_type, 'scatter_regression');
  });
});

describe('scatter chart validation', () => {
  it('accepts a risk-return scatter with two valid points and no optimizer', () => {
    const result = validateScatterChartPayload(baseScatterPayload());
    assert.equal(result.valid, true);
  });

  it('blocks scatter payloads with fewer than two valid points', () => {
    const result = validateScatterChartPayload(baseScatterPayload({
      point_count: 1,
      data: [{ ticker: 'AAPL', annualized_volatility_percent: 18.4, annualized_return_percent: 12.6, sector: 'Technology' }],
      series: [{ name: 'Technology', data: [{ id: 'AAPL', x: 18.4, y: 12.6 }] }],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /at least two valid points/);
  });

  it('blocks bubble scatter when size field is missing', () => {
    const result = validateScatterChartPayload(baseScatterPayload({
      plot_id: 'bubble_risk_return',
      chart_type: 'bubble_scatter',
      component: 'ScatterChartPro',
      size_axis: 'bubble_size_value',
      required_fields: ['ticker', 'annualized_volatility_percent', 'annualized_return_percent', 'bubble_size_value', 'sector'],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /bubble_size_value/);
  });

  it('blocks regression scatter with too few points', () => {
    const result = validateScatterChartPayload(baseScatterPayload({
      plot_id: 'scatter_with_regression_line',
      chart_type: 'scatter_regression',
      x_axis: 'x_return_percent',
      y_axis: 'y_return_percent',
      point_id: 'date',
      color_axis: null,
      required_fields: ['date', 'x_return_percent', 'y_return_percent'],
      point_count: 2,
      data: [
        { date: '2024-01-02', x_return_percent: 1, y_return_percent: 0.7 },
        { date: '2024-01-03', x_return_percent: -1, y_return_percent: -0.6 },
      ],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /at least three/);
  });

  it('blocks ownership overlap scatter without graph data', () => {
    const result = validateScatterChartPayload(baseScatterPayload({
      plot_id: 'ownership_overlap_correlation_scatter',
      x_axis: 'ownership_overlap',
      y_axis: 'return_correlation',
      point_id: 'ticker_pair',
      color_axis: null,
      required_fields: ['ticker_pair', 'ownership_overlap', 'return_correlation'],
      point_count: 2,
      data: [
        { ticker_pair: 'AAPL-MSFT', ownership_overlap: 0.2, return_correlation: 0.7 },
        { ticker_pair: 'AAPL-NVDA', ownership_overlap: 0.1, return_correlation: 0.6 },
      ],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /institutional graph data/);
  });

  it('blocks optimizer leakage on scatter diagnostics', () => {
    const result = validateScatterChartPayload(baseScatterPayload({ optimizer_called: true }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /optimizer/);
  });
});


