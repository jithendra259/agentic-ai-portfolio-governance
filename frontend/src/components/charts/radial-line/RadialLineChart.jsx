import React from 'react';
import { Box, Typography } from '@mui/material';
import { Unstable_RadialLineChart as RadialLineChart } from '@mui/x-charts-premium/RadialLineChart';
import { getResponsiveChartHeight } from './radialLineChartUtils';

export default function RadialLineChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const categories = spec.categories || [];
  const series = spec.series || [];
  
  if (!categories.length || !series.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No radial line data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadialLineChart 
        height={height} 
        series={series} 
        rotationAxis={[{ data: categories, scaleType: 'point' }]} 
        radiusAxis={[{ scaleType: 'linear' }]} 
        grid={spec.grid || { radius: true, rotation: true }} 
        hideLegend={spec.hideLegend ?? false} 
      />
    </Box>
  );
}
