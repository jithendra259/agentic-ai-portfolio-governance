import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { toFiniteNumber, getResponsiveChartHeight } from './sparklineChartUtils';

export default function SparklineChartRenderer({ spec }) {
  const chartProps = {};
  if (spec.plotType) chartProps.plotType = spec.plotType;
  if (spec.area) chartProps.area = true;
  if (spec.curve) chartProps.curve = spec.curve;
  if (spec.color) chartProps.color = spec.color;
  
  chartProps.showHighlight = spec.showHighlight !== undefined ? spec.showHighlight : true;
  chartProps.showTooltip = spec.showTooltip !== undefined ? spec.showTooltip : true;
  if (spec.baseline !== undefined) chartProps.baseline = spec.baseline;
  
  const height = getResponsiveChartHeight(spec, 64);
  
  const sparkData = useMemo(() => 
    (spec.data || [])
      .map((value) => toFiniteNumber(value))
      .filter((value) => Number.isFinite(value)), 
    [spec.data]
  );
  
  const xAxisConfig = useMemo(() => {
    if (!spec.xAxis) return undefined;
    const ax = { ...spec.xAxis };
    if (ax.data && Array.isArray(ax.data) && ax.scaleType === 'time') {
      ax.data = ax.data.map((d) => new Date(d));
    }
    return ax;
  }, [spec.xAxis]);
  
  const yAxisConfig = useMemo(() => (spec.yAxis ? { ...spec.yAxis } : undefined), [spec.yAxis]);

  return (
    <Box sx={{ width: '100%', minWidth: 220, display: 'flex', justifyContent: 'center', p: 0.5 }}>
      <SparkLineChart 
        data={sparkData} 
        height={height} 
        {...(xAxisConfig ? { xAxis: xAxisConfig } : {})} 
        {...(yAxisConfig ? { yAxis: yAxisConfig } : {})} 
        {...chartProps} 
      />
    </Box>
  );
}
