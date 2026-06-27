import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import RecessionBands from './RecessionBands';
import {
  AXIS_STYLE,
  GRID_STYLE,
  prepareDataset,
  prepareSeries,
  prepareYAxis,
  prepareMargins,
} from './lineChartUtils';
import { dateScaleType, formatChartDate } from '../../../utils/plotDataParser.js';

export default function LineChartRenderer({ spec }) {
  const { dataset, series, yAxisConfig, margins, pointCount } = useMemo(() => {
    if (!spec?.series?.length) {
      return {
        dataset: [],
        series: [],
        yAxisConfig: [],
        margins: { top: 24, right: 24, left: 60, bottom: 40 },
        pointCount: 0,
      };
    }
    const specText = `${spec?.title || ''} ${spec?.plot_id || ''} ${spec?.chart_type || ''}`.toLowerCase();
    const inferredArea = spec.area === true || specText.includes('area') || specText.includes('drawdown');

    const dataset = prepareDataset(spec.series, spec);
    const series = prepareSeries(spec, inferredArea);
    const yAxisConfig = prepareYAxis(spec);
    const margins = prepareMargins(spec);
    const pointCount = spec.series.reduce((total, item) => total + (Array.isArray(item?.data) ? item.data.length : 0), 0);

    return { dataset, series, yAxisConfig, margins, pointCount };
  }, [spec]);

  if (!dataset.length) return null;
  const gridConfig = spec.grid || { horizontal: true };
  const hasAreaSeries = series.some((entry) => entry.area);
  
  const animationSx = useMemo(() => (spec.animation ? {
    '& .MuiLineElement-root.MuiCharts-animate': { 
      animationDuration: spec.animation.duration || '1s', 
      animationDelay: spec.animation.delay || '0s', 
      animationTimingFunction: spec.animation.easing || 'ease-out' 
    },
    '& .MuiAreaElement-root.MuiCharts-animate': { 
      animationDuration: spec.animation.duration || '1s', 
      animationDelay: spec.animation.delay || '0s', 
      animationTimingFunction: spec.animation.easing || 'ease-out' 
    },
    '& .MuiAreaElement-root': { opacity: 0.35 },
    '& .MuiChartsGrid-line': GRID_STYLE,
  } : {
    '& .MuiAreaElement-root': { opacity: hasAreaSeries ? 0.35 : 0.0 },
  }), [spec.animation, hasAreaSeries]);

  const [chartRef, chartWidth] = useResponsiveChartWidth();
  const chartHeight = Number.isFinite(Number(spec.height)) ? Number(spec.height) : 320;
  const skipAnimation = Boolean(spec.skipAnimation || pointCount > 500);
  const xAxisConfig = useMemo(() => [{
    id: 'x-axis',
    dataKey: 'date',
    scaleType: dateScaleType(spec),
    tickLabelStyle: AXIS_STYLE,
    label: spec.x_label || 'Date',
    valueFormatter: (date) => formatChartDate(date, spec),
    ...(spec.zoom ? { zoom: spec.zoom } : {}),
  }], [spec]);
  const chartMargin = useMemo(() => margins, [margins]);

  return (
    <Box ref={chartRef} sx={{ width: '100%', height: chartHeight, minWidth: 0, ...animationSx }}>
      <LineChart
        width={chartWidth}
        height={chartHeight}
        dataset={dataset}
        series={series}
        xAxis={xAxisConfig}
        yAxis={yAxisConfig}
        margin={chartMargin}
        grid={gridConfig}
        skipAnimation={skipAnimation}
      >
        {spec.recessions && <RecessionBands periods={spec.recessions} />}
      </LineChart>
    </Box>
  );
}
