import React from 'react';
import { Box, Typography } from '@mui/material';
import { Unstable_RadialBarChart as RadialBarChart } from '@mui/x-charts-premium/RadialBarChart';

function getResponsiveChartHeight(spec, fallback = 360) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}

export default function RadialBarChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const categories = spec.categories || [];
  const series = spec.series || [];
  if (!categories.length || !series.length) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography variant="body2" sx={{ color: '#9ca3af' }}>No radial bar data available.</Typography></Box>;
  }
  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadialBarChart height={height} series={series} rotationAxis={[{ data: categories, scaleType: 'band' }]} radiusAxis={[{ scaleType: 'linear' }]} grid={spec.grid || { radius: true, rotation: true }} hideLegend={spec.hideLegend ?? false} />
    </Box>
  );
}
