import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { adaptBarChartPayload } from './barChartDataAdapter.js';
import { inferBarMode } from './barChartIntelligence.js';
import { CHART_TIER } from './chartTierConfig.js';
import { getBarChartDefinition } from './barChartRegistry.js';
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

  it('premium disabled range bars fall back to a standard min/max bar', () => {
    const adapted = adaptBarChartPayload({
      plot_type: 'bar',
      plot_id: 'return_range_by_ticker',
      chart_type: 'rangeBar',
      chart_tier: CHART_TIER.PREMIUM,
      component: 'BarChartPremium',
      requires_premium: true,
      fallback_chart: 'standard_min_max_bar',
      title: 'Return Range by Ticker',
      x_axis: 'ticker',
      y_axis: 'return_range_percent',
      unit: 'percent',
      series: [{ key: 'return_range', label: 'Return range' }],
      data: [{ ticker: 'AAPL', min_return: -2.3, max_return: 3.1 }],
    }, { premiumEnabled: false });

    assert.equal(adapted.valid, true);
    assert.equal(adapted.usePremiumRenderer, false);
    assert.equal(adapted.payload.fallback_used, true);
    assert.equal(adapted.payload.status, 'premium_unavailable');
    assert.deepEqual(adapted.series.map((series) => series.dataKey), ['min_return', 'max_return']);
  });

  it('premium enabled return range renders a rangeBar series', () => {
    const adapted = adaptBarChartPayload({
      plot_type: 'bar',
      plot_id: 'return_range_by_ticker',
      chart_type: 'rangeBar',
      chart_tier: CHART_TIER.PREMIUM,
      component: 'BarChartPremium',
      requires_premium: true,
      fallback_chart: 'standard_min_max_bar',
      title: 'Return Range by Ticker',
      x_axis: 'ticker',
      y_axis: 'return_range_percent',
      unit: 'percent',
      series: [{ key: 'return_range', label: 'Return range' }],
      data: [{ ticker: 'AAPL', min_return: -2.3, max_return: 3.1 }],
    }, { premiumEnabled: true });

    assert.equal(adapted.valid, true);
    assert.equal(adapted.usePremiumRenderer, true);
    assert.equal(adapted.series[0].type, 'rangeBar');
    assert.deepEqual(adapted.series[0].datasetKeys, { start: 'min_return', end: 'max_return' });
  });

  it('histogram payload keeps adjacent bars tight', () => {
    const adapted = adaptBarChartPayload({
      plot_type: 'bar',
      plot_id: 'return_distribution_histogram',
      chart_type: 'histogram',
      chart_tier: CHART_TIER.FREE,
      component: 'BarChart',
      title: 'Return Distribution',
      x_axis: 'return_bin',
      y_axis: 'count',
      unit: 'count',
      series: [{ key: 'count', label: 'Frequency' }],
      data: [{ bin_start: -2, bin_end: -1, count: 4 }],
    });

    assert.equal(adapted.valid, true);
    assert.equal(adapted.xAxis[0].categoryGapRatio, 0);
    assert.equal(adapted.xAxis[0].barGapRatio, 0);
    assert.equal(adapted.dataset[0].return_bin, '-2.0--1.0');
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

  it('registry marks premium chart definitions as optional', () => {
    const definition = getBarChartDefinition('return_range_by_ticker');
    assert.equal(definition.chart_tier, CHART_TIER.PREMIUM);
    assert.equal(definition.requires_premium, true);
    assert.equal(definition.fallback_chart, 'standard_min_max_bar');
  });

  it('range bars require start and end fields', () => {
    const result = validateBarChartPayload({
      ...basePayload({
        plot_id: 'return_range_by_ticker',
        chart_type: 'rangeBar',
        chart_tier: CHART_TIER.PREMIUM,
        component: 'BarChartPremium',
        requires_premium: true,
        fallback_chart: 'standard_min_max_bar',
        x_axis: 'ticker',
        y_axis: 'return_range_percent',
        data: [{ ticker: 'AAPL', min_return: -2.3 }],
      }),
    }, { premiumEnabled: true });
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /max_return/);
  });

  it('waterfall charts require start_value and end_value', () => {
    const result = validateBarChartPayload(basePayload({
      plot_id: 'portfolio_return_waterfall',
      chart_type: 'rangeBar',
      chart_tier: CHART_TIER.PREMIUM,
      component: 'BarChartPremium',
      requires_premium: true,
      fallback_chart: 'contribution_bar',
      x_axis: 'component',
      y_axis: 'return_contribution_percent',
      data: [{ component: 'AAPL', start_value: 0 }],
    }), { premiumEnabled: true });
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /end_value/);
  });

  it('mirrored current vs advisory blocks missing advisory weights', () => {
    const result = validateBarChartPayload(basePayload({
      plot_id: 'current_vs_advisory_mirrored_bar',
      chart_type: 'mirroredBar',
      chart_tier: CHART_TIER.PREMIUM,
      component: 'BarChartPremium',
      requires_premium: true,
      fallback_chart: 'grouped_bar',
      x_axis: 'weight_percent',
      y_axis: 'ticker',
      data: [{ ticker: 'AAPL', current_weight: 20 }],
    }), { premiumEnabled: true });
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /advisory_weight/);
  });

  it('free charts remain valid when premium is disabled', () => {
    const result = validateBarChartPayload(basePayload(), { premiumEnabled: false });
    assert.equal(result.valid, true);
  });
});
