import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getPieChartDefinition } from './pieChartRegistry.js';
import { validatePieChartPayload } from './pieChartValidator.js';

function basePiePayload(overrides = {}) {
  return {
    plot_type: 'pie',
    plot_id: 'sector_allocation_donut',
    chart_type: 'donut',
    chart_tier: 'free',
    component: 'PieChart',
    title: 'Sector Allocation Donut',
    category_field: 'sector',
    value_field: 'weight_percent',
    unit: '%',
    slice_count: 2,
    total_value: 100,
    required_fields: ['sector', 'weight_percent'],
    optimizer_called: false,
    advisory_allocation_generated: false,
    data: [
      { sector: 'Technology', weight_percent: 70 },
      { sector: 'Financials', weight_percent: 30 },
    ],
    series: [
      {
        name: 'sector_weight_percent',
        data: [
          { id: 'Technology', label: 'Technology', value: 70 },
          { id: 'Financials', label: 'Financials', value: 30 },
        ],
      },
    ],
    ...overrides,
  };
}

describe('pie chart registry', () => {
  it('registers portfolio governance pie chart families', () => {
    assert.equal(getPieChartDefinition('sector_allocation_donut').chart_type, 'donut');
    assert.equal(getPieChartDefinition('sector_ticker_nested_donut').chart_type, 'nested_donut');
    assert.equal(getPieChartDefinition('semi_donut_risk_gauge').chart_type, 'semi_donut');
  });
});

describe('pie chart validation', () => {
  it('accepts sector allocation donut values that sum to 100', () => {
    const result = validatePieChartPayload(basePiePayload());
    assert.equal(result.valid, true);
  });

  it('blocks missing category/value fields', () => {
    const result = validatePieChartPayload(basePiePayload({
      data: [{ sector: 'Technology' }],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /weight_percent/);
  });

  it('blocks negative values', () => {
    const result = validatePieChartPayload(basePiePayload({
      data: [
        { sector: 'Technology', weight_percent: 105 },
        { sector: 'Financials', weight_percent: -5 },
      ],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /negative/);
  });

  it('blocks allocation percentages that do not sum near 100', () => {
    const result = validatePieChartPayload(basePiePayload({
      data: [
        { sector: 'Technology', weight_percent: 60 },
        { sector: 'Financials', weight_percent: 20 },
      ],
      total_value: 80,
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /sum close to 100/);
  });

  it('blocks ticker donuts above the readability slice limit', () => {
    const result = validatePieChartPayload(basePiePayload({
      plot_id: 'ticker_allocation_donut',
      category_field: 'ticker',
      value_field: 'weight_percent',
      required_fields: ['ticker', 'weight_percent'],
      slice_count: 9,
      data: Array.from({ length: 9 }, (_, index) => ({ ticker: `T${index + 1}`, weight_percent: 100 / 9 })),
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /max_slices/);
  });

  it('requires parent-child fields for nested donuts', () => {
    const result = validatePieChartPayload(basePiePayload({
      plot_id: 'sector_ticker_nested_donut',
      chart_type: 'nested_donut',
      category_field: 'ticker',
      value_field: 'ticker_weight_percent',
      required_fields: ['sector', 'sector_weight_percent', 'ticker', 'ticker_weight_percent'],
      data: [{ sector: 'Technology', sector_weight_percent: 100, ticker_weight_percent: 100 }],
      series: [{ name: 'sectors', data: [] }],
    }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /ticker/);
    assert.match(result.errors.join('\n'), /two series/);
  });

  it('blocks optimizer leakage on pie-only plots', () => {
    const result = validatePieChartPayload(basePiePayload({ optimizer_called: true }));
    assert.equal(result.valid, false);
    assert.match(result.errors.join('\n'), /optimizer/);
  });
});


