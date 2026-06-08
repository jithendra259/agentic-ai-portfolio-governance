import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { ScatterChart } from '@mui/x-charts/ScatterChart';
import { ScatterChartPro } from '@mui/x-charts-pro/ScatterChartPro';
import { ScatterChartPremium } from '@mui/x-charts-premium/ScatterChartPremium';
import { useXScale, useYScale } from '@mui/x-charts/hooks';
import { ChartsClipPath } from '@mui/x-charts/ChartsClipPath';

const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };
const GRID_STYLE = { stroke: '#374151', strokeDasharray: '4 4' };

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getValueFormatter(format) {
  if (format === 'percent' || format === '%') return (v) => (v == null ? '' : `${v.toFixed(1)}%`);
  if (format === 'decimal') return (v) => (v == null ? '' : Number(v).toFixed(2));
  if (format === 'beta') return (v) => (v == null ? '' : `${Number(v).toFixed(2)} beta`);
  if (format === 'k') return (v) => (v == null ? '' : `${(v / 1000).toFixed(1)}k`);
  if (format === 'currency') return (v) => (v == null ? '' : `$${v.toLocaleString()}`);
  return (v) => {
    if (v == null) return '';
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return Number(v).toFixed(2);
  };
}

function useResponsiveChartWidth(fallback = 360) {
  const ref = React.useRef(null);
  const [width, setWidth] = React.useState(fallback);
  React.useEffect(() => {
    if (!ref.current) return;
    const updateWidth = (value) => setWidth(Math.max(280, Math.floor(value || fallback)));
    updateWidth(ref.current.getBoundingClientRect().width);
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) updateWidth(entries[0].contentRect.width);
    });
    resizeObserver.observe(ref.current);
    return () => resizeObserver.disconnect();
  }, [fallback]);
  return [ref, width];
}

function ScatterRegressionLine({ regression }) {
  const xScale = useXScale();
  const yScale = useYScale();
  const clipPathId = `scatter-regression-${React.useId()}`;
  if (!regression || !xScale || !yScale) return null;
  const xMin = Number(regression.x_min);
  const xMax = Number(regression.x_max);
  const slope = Number(regression.slope);
  const intercept = Number(regression.intercept);
  const yMin = Number.isFinite(Number(regression.y_min)) ? Number(regression.y_min) : slope * xMin + intercept;
  const yMax = Number.isFinite(Number(regression.y_max)) ? Number(regression.y_max) : slope * xMax + intercept;
  if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) return null;
  const x1 = xScale(xMin); const x2 = xScale(xMax); const y1 = yScale(yMin); const y2 = yScale(yMax);
  if (![x1, x2, y1, y2].every(Number.isFinite)) return null;
  return (
    <React.Fragment>
      <ChartsClipPath id={clipPathId} />
      <g clipPath={`url(#${clipPathId})`}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={regression.color || '#f25467'} strokeWidth={2} strokeDasharray={regression.strokeDasharray || '6 4'} pointerEvents="none" />
      </g>
    </React.Fragment>
  );
}

export default function ScatterChartRenderer({ spec }) {
  const series = useMemo(() => {
    if (!spec?.series || !Array.isArray(spec.series)) return [];
    return spec.series.map((s, i) => {
      const dataPoints = (s.data || []).map((pt, j) => ({
        x: toFiniteNumber(pt.x),
        y: toFiniteNumber(pt.y),
        id: pt.id !== undefined ? pt.id : `pt-${j}`,
        label: pt.label,
        ...(pt.z !== undefined ? { z: toFiniteNumber(pt.z) } : {}),
        ...(pt.sizeValue !== undefined ? { sizeValue: toFiniteNumber(pt.sizeValue) } : {}),
        ...(pt.colorValue !== undefined ? { colorValue: pt.colorValue } : {}),
      }));
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
  }, [spec]);

  const xAxisConfig = useMemo(() => {
    if (spec.xAxis && Array.isArray(spec.xAxis)) {
      return spec.xAxis.map((ax) => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: ax.valueFormatter || getValueFormatter(ax.value_format || spec.x_format || spec.x_unit),
        label: ax.label || spec.x_label || spec.x_axis || '',
        domainLimit: ax.domainLimit || 'nice',
      }));
    }
    return [{ tickLabelStyle: AXIS_STYLE, label: spec.x_label || '', valueFormatter: getValueFormatter(spec.x_format || spec.x_unit), domainLimit: 'nice' }];
  }, [spec]);

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
  const [chartRef, chartWidth] = useResponsiveChartWidth();
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
