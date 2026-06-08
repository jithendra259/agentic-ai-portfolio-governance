import React from 'react';
import { Box, Typography } from '@mui/material';
import { Heatmap } from '@mui/x-charts-premium/Heatmap';

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getResponsiveChartHeight(spec, fallback = 420) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}

const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };

export default function HeatmapChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 420);
  const xAxis = spec.xAxis || [{ data: spec.xLabels || [] }];
  const yAxis = spec.yAxis || [{ data: spec.yLabels || [] }];
  const series = (spec.series || []).map((serie, index) => ({ id: serie.id || `heatmap-${index}`, label: serie.label || spec.title || 'Heatmap', data: serie.data || [], valueFormatter: (value) => value == null ? '' : toFiniteNumber(value).toFixed(3), ...serie }));
  if (!series.length || !series.some((serie) => Array.isArray(serie.data) && serie.data.length)) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography variant="body2" sx={{ color: '#9ca3af' }}>No heatmap data available.</Typography></Box>;
  }
  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <Heatmap height={height} xAxis={xAxis} yAxis={yAxis} series={series} borderRadius={spec.borderRadius ?? 4} hideLegend={spec.hideLegend ?? false} margin={spec.margin || { top: 24, right: 36, bottom: 58, left: 72 }} sx={{ '& .MuiChartsAxis-tickLabel': AXIS_STYLE, '& .MuiChartsLegend-root': { color: '#d1d5db' } }} />
    </Box>
  );
}
