import React from 'react';
import { Box, Typography } from '@mui/material';
import { FunnelChart } from '@mui/x-charts-premium/FunnelChart';
import { getResponsiveChartHeight } from './funnelChartUtils';

export default function FunnelChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const series = spec.series || [];
  
  if (!series.length || !series.some((serie) => Array.isArray(serie.data) && serie.data.length)) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No funnel data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <FunnelChart 
        height={height} 
        series={series} 
        hideLegend={spec.hideLegend ?? false} 
        margin={spec.margin || { top: 24, right: 24, bottom: 30, left: 24 }} 
      />
    </Box>
  );
}
