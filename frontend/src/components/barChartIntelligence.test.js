import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { adaptBarChartPayload } from './barChartDataAdapter.js';
import { inferBarMode } from './barChartIntelligence.js';
import { validateBarChartPayload } from './barChartValidator.js';

function basePayload(overrides = {}) {
  return {
    plot_type: 'bar',
    plot_id: 'ticker_concentration_plot',
    chart_type: 'bar',
    bar_mode: 'vertical',
    title: 'Ticker Concentration',
    universe: 'U1',
    analysis_id: 'A1',
    x_axis: 'allocation_percent',
    y_axis: 'ticker',
    unit: 'percent',
    sort: 'descending',
    series: [{ key: 'allocation_percent', label: 'Current allocation' }],
    data: [{ ticker: 'AAPL', allocation_percent: 15 }],
    thresholds: [{ name: 'Max ticker cap', value: 20 }],
    ...overrides,
  };
}

describe('bar chart intelligence', () => {
  it('ticker concentration with 20 tickers uses horizontal layout', () => {
    const payload = basePayload({
      data: Array.from({ length: 20 }, (_, index) => ({
        ticker: `T${index + 1}`,
        allocation_percent: 20 - index,
      })),
    });
    const adapted = adaptBarChartPayload(payload);
    assert.equal(adapted.layout, 'horizontal');
  });

  it('literal percent unit formats bar values as percentages', () => {
    const adapted = adaptBarChartPayload(basePayload({ unit: '%' }));
    assert.equal(adapted.series[0].valueFormatter(5), '5.0%');
  });

  it('sector concentration with one sector renders as valid bar data', () => {
    const adapted = adaptBarChartPayload(basePayload({
      plot_id: 'sector_concentration_plot',
      title: 'Sector Concentration',
      x_axis: 'allocation_percent',
      y_axis: 'sector',
      data: [{ sector: 'Technology', allocation_percent: 100 }],
    }));
    assert.equal(adapted.valid, true);
    assert.equal(adapted.dataset[0].sector, 'Technology');
  });

  it('current vs advisory allocation uses grouped mode', () => {
    const payload = basePayload({
      plot_id: 'current_vs_advisory_allocation_by_ticker',
      title: 'Current vs Advisory Allocation by Ticker',
      x_axis: 'ticker',
      y_axis: 'allocation_percent',
      series: [
        { key: 'current_allocation_percent', label: 'Current allocation' },
        { key: 'advisory_allocation_percent', label: 'Suggested exposure weights' },
      ],
      data: [{ ticker: 'AAPL', current_allocation_percent: 20, advisory_allocation_percent: 14 }],
    });
    assert.equal(inferBarMode(payload), 'grouped');
    assert.equal(adaptBarChartPayload(payload).series.length, 2);
  });

  it('allocation change supports positive and negative values', () => {
    const adapted = adaptBarChartPayload(basePayload({
      plot_id: 'allocation_change_by_ticker',
      title: 'Allocation Change by Ticker',
      x_axis: 'ticker',
      y_axis: 'allocation_change_percent',
      series: [{ key: 'allocation_change_percent', label: 'Suggested exposure change' }],
      data: [
        { ticker: 'AAPL', allocation_change_percent: 3 },
        { ticker: 'MSFT', allocation_change_percent: -2 },
      ],
    }));
    assert.equal(adapted.dataset.some((row) => row.allocation_change_percent < 0), true);
    assert.equal(adapted.dataset.some((row) => row.allocation_change_percent > 0), true);
  });

  it('risk contribution sorts descending', () => {
    const adapted = adaptBarChartPayload(basePayload({
      plot_id: 'risk_contribution_by_ticker',
      title: 'Risk Contribution by Ticker',
      x_axis: 'risk_contribution_percent',
      y_axis: 'ticker',
      series: [{ key: 'risk_contribution_percent', label: 'Risk contribution' }],
      data: [
        { ticker: 'AAPL', risk_contribution_percent: 12 },
        { ticker: 'NVDA', risk_contribution_percent: 31 },
        { ticker: 'MSFT', risk_contribution_percent: 18 },
      ],
    }));
    assert.deepEqual(adapted.dataset.map((row) => row.ticker), ['NVDA', 'MSFT', 'AAPL']);
  });
});

describe('bar chart validation', () => {
  it('HHI and effective holdings formulas are checked separately from sector HHI', () => {
    const result = validateBarChartPayload(basePayload({
      metrics: {
        ticker_hhi: 0.05,
        ticker_effective_holdings: 20,
        sector_hhi: 1,
        sector_effective_sectors: 1,
      },
    }));
    assert.equal(result.valid, true);
  });

  it('blocks requested plot_id mismatch', () => {
    const result = validateBarChartPayload(basePayload(), { requestedPlotId: 'sector_concentration_plot' });
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /does not match/);
  });

  it('missing required fields produce validation errors instead of renderer crashes', () => {
    const result = validateBarChartPayload(basePayload({
      data: [{ ticker: 'AAPL' }],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /allocation_percent/);
  });

  it('U1 plot does not silently accept a U3 universe', () => {
    const result = validateBarChartPayload(basePayload({ universe: 'U3' }), { universe: 'U1' });
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /universe/);
  });
});
