import React from 'react';
import { Box, Typography } from '@mui/material';
import { RadarChart } from '@mui/x-charts-premium/RadarChart';

function getResponsiveChartHeight(spec, fallback = 360) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}

export default function RadarChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const metrics = spec.radar?.metrics || spec.metrics || [];
  const series = spec.series || [];
  if (!metrics.length || !series.length) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography variant="body2" sx={{ color: '#9ca3af' }}>No radar data available.</Typography></Box>;
  }
  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadarChart height={height} radar={{ metrics }} series={series} hideLegend={spec.hideLegend ?? false} margin={spec.margin || { top: 24, right: 28, bottom: 30, left: 28 }} />
    </Box>
  );
}
