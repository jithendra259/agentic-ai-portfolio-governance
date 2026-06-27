import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { BarChart } from '@mui/x-charts/BarChart';
import { BarChartPremium } from '@mui/x-charts-premium/BarChartPremium';
import { PiecewiseColorLegend } from '@mui/x-charts/ChartsLegend';
import { adaptBarChartPayload } from './barChartDataAdapter.js';
import { chooseRenderer } from './barChartIntelligence.js';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import { AXIS_STYLE, GRID_STYLE, decorateAxes } from './barChartUtils';
import BarThresholdNotes from './BarThresholdNotes';
import BarWarnings from './BarWarnings';

export default function SmartBarChartRenderer({ spec }) {
  const adapted = useMemo(() => adaptBarChartPayload(spec), [spec]);
  // SmartBarChartRenderer uses a minWidth of 300px
  const [chartRef, chartWidth] = useResponsiveChartWidth(360, 300);

  const isPiecewise = useMemo(() => {
    const checkAxis = (axis) => axis?.some((ax) => ax?.colorMap?.type === 'piecewise');
    return checkAxis(adapted.xAxis) || checkAxis(adapted.yAxis);
  }, [adapted.xAxis, adapted.yAxis]);

  const slotsConfig = useMemo(() => {
    return isPiecewise ? { legend: PiecewiseColorLegend } : undefined;
  }, [isPiecewise]);

  if (!adapted.valid) {
    return (
      <Box sx={{ p: 2, minHeight: 180, display: 'grid', placeItems: 'center', color: '#9ca3af' }}>
        <Typography variant="body2">{adapted.reason || 'No bar chart data available.'}</Typography>
      </Box>
    );
  }

  const renderer = adapted.renderer || chooseRenderer(adapted.payload);
  const ChartComponent = adapted.usePremiumRenderer ? BarChartPremium : BarChart;
  const animationSx = adapted.animation ? {
    '& .MuiBarElement-root': {
      animationDuration: adapted.animation.duration || '800ms',
      animationDelay: adapted.animation.delay || '0s',
      animationTimingFunction: adapted.animation.easing || 'ease-out',
    },
  } : {};

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <ChartComponent
        width={chartWidth}
        height={adapted.chartHeight}
        dataset={adapted.dataset}
        series={adapted.series}
        xAxis={decorateAxes(adapted.xAxis)}
        yAxis={decorateAxes(adapted.yAxis)}
        layout={adapted.layout}
        renderer={renderer}
        margin={adapted.margin}
        grid={adapted.grid}
        borderRadius={adapted.borderRadius}
        skipAnimation={adapted.skipAnimation}
        slots={slotsConfig}
        slotProps={{
          legend: {
            position: { vertical: 'top', horizontal: 'middle' },
            sx: { color: '#e5e7eb', fontSize: 12, fontWeight: 600 },
          },
        }}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-label': { fill: '#e5e7eb', fontSize: 12, fontWeight: 700 },
          '& .MuiChartsGrid-line': GRID_STYLE,
          '& .MuiBarLabel-root': { fill: '#fff', fontSize: 12, fontWeight: 800 },
          '& .MuiChartsLegend-label': { color: '#e5e7eb !important' },
          ...animationSx,
        }}
      />
      <BarThresholdNotes thresholds={adapted.thresholds} />
      <BarWarnings warnings={adapted.warnings} interpretation={adapted.interpretation} />
    </Box>
  );
}
