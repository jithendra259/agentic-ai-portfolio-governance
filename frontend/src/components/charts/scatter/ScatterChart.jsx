import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { ScatterChart } from '@mui/x-charts/ScatterChart';
import { ScatterChartPro } from '@mui/x-charts-pro/ScatterChartPro';
import { ScatterChartPremium } from '@mui/x-charts-premium/ScatterChartPremium';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import {
  PALETTE,
  AXIS_STYLE,
  GRID_STYLE,
  toFiniteNumber,
  getValueFormatter,
} from './scatterChartUtils';
import { dateParseMode, dateScaleType, formatChartDate, parseDateValue } from '../../../utils/plotDataParser.js';
import ScatterRegressionLine from './ScatterRegressionLine';

export default function ScatterChartRenderer({ spec }) {
  const xIsTime = spec?.x_type === 'time' || spec?.x_type === 'utc' || spec?.x_type === 'date_only' || spec?.x_scale === 'time';
  const xDateMode = dateParseMode(spec);
  const series = useMemo(() => {
    if (!spec?.series || !Array.isArray(spec.series)) return [];
    return spec.series.map((s, i) => {
      const dataPoints = (s.data || [])
        .map((pt, j) => {
          const x = xIsTime ? parseDateValue(pt.x ?? pt.date, { mode: xDateMode }) : toFiniteNumber(pt.x);
          const y = toFiniteNumber(pt.y);
          if (x == null || y == null) return null;
          return {
            x,
            y,
            id: pt.id !== undefined ? pt.id : `pt-${j}`,
            label: pt.label,
            ...(pt.z !== undefined ? { z: toFiniteNumber(pt.z) } : {}),
            ...(pt.sizeValue !== undefined ? { sizeValue: toFiniteNumber(pt.sizeValue) } : {}),
            ...(pt.colorValue !== undefined ? { colorValue: pt.colorValue } : {}),
          };
        })
        .filter(Boolean);
      const entry = {
        id: s.id || s.name || `scatter-${i}`,
        type: 'scatter',
        label: s.label || s.name || `Series ${i + 1}`,
        data: dataPoints,
        color: s.color || PALETTE[i % PALETTE.length],
      };
      if (s.markerSize !== undefined) entry.markerSize = s.markerSize;
      if (s.sizeAxisId) entry.sizeAxisId = s.sizeAxisId;
      if (s.colorAxisId) entry.colorAxisId = s.colorAxisId;
      if (s.labelMarkType) entry.labelMarkType = s.labelMarkType;
      if (s.highlightScope) entry.highlightScope = s.highlightScope;
      else if (spec.highlightScope) entry.highlightScope = spec.highlightScope;
      return entry;
    });
  }, [spec, xIsTime]);

  const xAxisConfig = useMemo(() => {
    if (spec.xAxis && Array.isArray(spec.xAxis)) {
      return spec.xAxis.map((ax) => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: ax.valueFormatter || getValueFormatter(ax.value_format || spec.x_format || spec.x_unit),
        label: ax.label || spec.x_label || spec.x_axis || '',
        scaleType: ax.scaleType || (xIsTime ? dateScaleType(spec) : undefined),
        domainLimit: ax.domainLimit || 'nice',
      }));
    }
    return [{
      tickLabelStyle: AXIS_STYLE,
      label: spec.x_label || '',
      valueFormatter: xIsTime
        ? (date) => formatChartDate(date, spec)
        : getValueFormatter(spec.x_format || spec.x_unit),
      scaleType: xIsTime ? dateScaleType(spec) : undefined,
      domainLimit: 'nice',
    }];
  }, [spec, xIsTime]);

  const yAxisConfig = useMemo(() => {
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      return spec.yAxis.map((ax) => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: ax.valueFormatter || getValueFormatter(ax.value_format || spec.y_format || spec.y_unit),
        label: ax.label || spec.y_label || spec.y_axis || '',
        domainLimit: ax.domainLimit || 'nice',
      }));
    }
    return [{ tickLabelStyle: AXIS_STYLE, label: spec.y_label || '', valueFormatter: getValueFormatter(spec.y_format || spec.y_unit), domainLimit: 'nice' }];
  }, [spec]);

  const zAxisConfig = useMemo(() => (spec.zAxis && Array.isArray(spec.zAxis) ? spec.zAxis.map((ax) => ({ ...ax })) : undefined), [spec]);
  
  const chartProps = {};
  if (spec.skipAnimation) chartProps.skipAnimation = true;
  if (spec.hideLegend) chartProps.hideLegend = true;
  if (spec.colors && Array.isArray(spec.colors)) chartProps.colors = spec.colors;
  if (spec.renderer) chartProps.renderer = spec.renderer;
  chartProps.hitAreaRadius = spec.hitAreaRadius !== undefined ? spec.hitAreaRadius : 20;
  
  const gridConfig = spec.grid || { horizontal: true, vertical: true };
  const [chartRef, chartWidth] = useResponsiveChartWidth(360, 280);
  const chartHeight = spec.height || 420;
  
  const ChartComponent = spec.component === 'ScatterChartPremium' || spec.chart_type === 'webgl_scatter' || spec.renderer === 'webgl'
    ? ScatterChartPremium
    : spec.component === 'ScatterChartPro' || spec.chart_type === 'bubble_scatter'
      ? ScatterChartPro
      : ScatterChart;

  if (!series.length) return null;

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <ChartComponent
        width={chartWidth}
        series={series}
        xAxis={xAxisConfig}
        yAxis={yAxisConfig}
        {...(zAxisConfig ? { zAxis: zAxisConfig } : {})}
        height={chartHeight}
        margin={spec.margin || { top: 46, right: 28, left: 72, bottom: 68 }}
        grid={gridConfig}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-label': { fill: '#e5e7eb', fontSize: 12, fontWeight: 600 },
          '& .MuiChartsGrid-line': GRID_STYLE,
          '& .MuiChartsLegend-root': { color: '#e5e7eb', fontSize: 12 },
          '& .MuiScatter-root .MuiScatter-mark': { fillOpacity: spec.chart_type === 'bubble_scatter' ? 0.68 : 0.9, strokeWidth: 1.2 },
        }}
        slotProps={{ legend: { position: { vertical: 'top', horizontal: 'middle' }, sx: { color: '#e5e7eb', fontSize: 12 } } }}
        {...chartProps}
      >
        {spec.regression_line && <ScatterRegressionLine regression={spec.regression_line} />}
      </ChartComponent>
    </Box>
  );
}
