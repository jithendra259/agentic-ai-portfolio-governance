import React, { useMemo, useCallback } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import { useChartConfig, useValueFormatter, useChartSlotProps } from '../common/useChartConfig';
import { useResponsiveChartDimensions, useAdaptiveMargins } from '../common/useResponsiveSizing';

/**
 * Enhanced LineChart component with improvements:
 * - Better responsive design
 * - Multiple axis support
 * - Enhanced accessibility
 * - Improved tooltips with custom formatting
 * - Area chart support
 * - Animation control
 * - Loading and error states
 */
export default function EnhancedLineChart({
  spec = {},
  loading = false,
  error = null,
  onItemClick = null,
  onHighlightChange = null
}) {
  const [chartRef, chartWidth] = useResponsiveChartWidth(500, 300);
  const config = useChartConfig(spec);
  const { formatValue } = useValueFormatter(spec.valueFormatter || {});
  const slotProps = useChartSlotProps(config.theme);

  // Calculate dimensions
  const dimensions = useResponsiveChartDimensions(chartWidth, spec.height || 380);
  const margins = useAdaptiveMargins({
    hasLongLabels: spec.hasLongLabels,
    hasMultilineTitle: spec.title && spec.title.length > 30,
    isVerticalBar: false,
    hasLegend: config.showLegend
  });

  // Process and validate data
  const chartData = useMemo(() => {
    if (!spec.series || !Array.isArray(spec.series) || spec.series.length === 0) {
      return { valid: false, reason: 'No data available' };
    }

    try {
      // Create dataset from series
      const dataset = [];
      const dateSet = new Set();

      spec.series.forEach(s => {
        if (Array.isArray(s.data)) {
          s.data.forEach((point, idx) => {
            const date = point.date || point.x || `Point ${idx}`;
            dateSet.add(date);
            if (!dataset.find(d => d.date === date)) {
              dataset.push({ date });
            }
          });
        }
      });

      // Merge data
      spec.series.forEach((s, seriesIdx) => {
        if (Array.isArray(s.data)) {
          s.data.forEach(point => {
            const date = point.date || point.x;
            const dataEntry = dataset.find(d => d.date === date);
            if (dataEntry) {
              dataEntry[`series_${seriesIdx}`] = point.value || point.y;
            }
          });
        }
      });

      // Create series config
      const series = spec.series.map((s, idx) => ({
        dataKey: `series_${idx}`,
        label: s.label || `Series ${idx + 1}`,
        color: s.color,
        area: s.area === true,
        type: 'line',
        curve: s.curve || 'linear',
        showMark: s.showMark !== false,
        valueFormatter: s.valueFormatter ? (v) => formatValue(v) : undefined
      }));

      // X-axis config
      const xAxisType = spec.xAxisType || (spec.series[0]?.data[0]?.date ? 'time' : 'point');
      const xAxis = [{
        id: 'x-axis',
        dataKey: 'date',
        scaleType: xAxisType,
        label: spec.xLabel || 'Date',
        tickLabelStyle: {
          fontSize: dimensions.isMobile ? 10 : 12,
          fill: config.theme === 'dark' ? '#e5e7eb' : '#374151'
        }
      }];

      // Y-axis config
      const yAxis = spec.yAxis ? [spec.yAxis] : [{
        id: 'y-axis',
        label: spec.yLabel || 'Value',
        scaleType: spec.yAxisScale || 'linear',
        tickLabelStyle: {
          fontSize: dimensions.isMobile ? 10 : 12,
          fill: config.theme === 'dark' ? '#e5e7eb' : '#374151'
        }
      }];

      return {
        valid: true,
        dataset,
        series,
        xAxis,
        yAxis
      };
    } catch (err) {
      console.error('Error processing line chart data:', err);
      return { valid: false, reason: 'Invalid data format' };
    }
  }, [spec, dimensions.isMobile, config.theme, formatValue]);

  // Custom tooltip formatter
  const tooltipFormatter = useCallback((params) => {
    if (!params || !Array.isArray(params)) return '';
    
    return params
      .map(p => `${p.seriesName}: ${formatValue(p.value)}`)
      .join('<br/>');
  }, [formatValue]);

  // Custom styling
  const chartSx = useMemo(() => ({
    '& .MuiChartsAxis-tickLabel': {
      fontSize: `${dimensions.isMobile ? 10 : 12}px`,
      fill: config.theme === 'dark' ? '#e5e7eb' : '#374151'
    },
    '& .MuiChartsAxis-label': {
      fill: config.theme === 'dark' ? '#d1d5db' : '#4b5563',
      fontSize: '12px',
      fontWeight: 700
    },
    '& .MuiChartsGrid-line': {
      stroke: config.gridStyle.stroke,
      strokeDasharray: '4 4'
    },
    '& .MuiLineElement-root': {
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      transition: 'all 0.2s ease-in-out',
      '&:hover': {
        filter: 'brightness(1.15)',
        cursor: 'pointer'
      }
    },
    '& .MuiAreaElement-root': {
      opacity: 0.3,
      transition: 'all 0.2s ease-in-out'
    },
    '& .MuiMarkElement-root': {
      transition: 'all 0.2s ease-in-out'
    },
    '& .MuiChartsLegend-label': {
      color: config.theme === 'dark' ? '#e5e7eb' : '#374151',
      fontSize: '12px',
      fontWeight: 500
    },
    width: '100%',
    minWidth: 0
  }), [dimensions.isMobile, config]);

  // Loading state
  if (loading) {
    return (
      <Box
        ref={chartRef}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: config.height,
          width: '100%'
        }}
      >
        <CircularProgress size={40} />
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Box
        ref={chartRef}
        sx={{
          p: 2,
          minHeight: config.height,
          display: 'grid',
          placeItems: 'center',
          color: '#ef4444'
        }}
      >
        <Typography variant="body2" color="error">
          {error}
        </Typography>
      </Box>
    );
  }

  // No data state
  if (!chartData.valid) {
    return (
      <Box
        ref={chartRef}
        sx={{
          p: 2,
          minHeight: config.height,
          display: 'grid',
          placeItems: 'center',
          color: '#9ca3af'
        }}
      >
        <Typography variant="body2">{chartData.reason}</Typography>
      </Box>
    );
  }

  return (
    <Box
      ref={chartRef}
      sx={{ width: '100%', minWidth: 0, pb: 1 }}
      role="img"
      aria-label={spec.title || 'Line chart'}
    >
      <LineChart
        width={dimensions.width}
        height={dimensions.height}
        dataset={chartData.dataset}
        series={chartData.series}
        xAxis={chartData.xAxis}
        yAxis={chartData.yAxis}
        margin={margins}
        grid={config.showGrid ? { vertical: true, horizontal: true } : undefined}
        slotProps={slotProps}
        sx={chartSx}
        {...config.animation}
        onItemClick={onItemClick}
        onHighlightChange={onHighlightChange}
        skipAnimation={spec.skipAnimation === true}
      />
    </Box>
  );
}
