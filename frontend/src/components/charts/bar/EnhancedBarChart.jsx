import React, { useMemo } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { BarChart } from '@mui/x-charts/BarChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import { useChartConfig, useValueFormatter, useChartSlotProps } from '../common/useChartConfig';
import { useResponsiveChartDimensions, useAdaptiveMargins } from '../common/useResponsiveSizing';

/**
 * Enhanced BarChart component with built-in improvements:
 * - Better responsive design with breakpoint awareness
 * - Enhanced accessibility
 * - Improved tooltips and interactions
 * - Better error handling and loading states
 * - Optimized animations
 * - Value formatting support
 */
export default function EnhancedBarChart({
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
  
  // Calculate dimensions based on width
  const dimensions = useResponsiveChartDimensions(chartWidth, spec.height || 320);
  const margins = useAdaptiveMargins({
    hasLongLabels: spec.hasLongLabels,
    hasMultilineTitle: spec.title && spec.title.length > 30,
    isVerticalBar: spec.layout !== 'horizontal',
    hasLegend: config.showLegend
  });

  // Process and validate data
  const chartData = useMemo(() => {
    if (!spec.series || !Array.isArray(spec.series) || spec.series.length === 0) {
      return { valid: false, reason: 'No data available' };
    }

    try {
      // Transform series data for MUI X Charts
      const series = spec.series.map(s => ({
        ...s,
        data: Array.isArray(s.data) ? s.data : [],
        type: 'bar'
      }));

      const xAxis = spec.xAxis || [];
      const yAxis = spec.yAxis || [];

      return {
        valid: true,
        series,
        xAxis,
        yAxis,
        dataset: spec.dataset || undefined
      };
    } catch (err) {
      console.error('Error processing bar chart data:', err);
      return { valid: false, reason: 'Invalid data format' };
    }
  }, [spec]);

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
    '& .MuiBarElement-root': {
      transition: 'all 0.2s ease-in-out',
      '&:hover': {
        filter: 'brightness(1.1)',
        cursor: 'pointer'
      }
    },
    '& .MuiChartsLegend-label': {
      color: config.theme === 'dark' ? '#e5e7eb' : '#374151',
      fontSize: '12px',
      fontWeight: 500
    },
    width: '100%',
    minWidth: 0
  }), [dimensions, config]);

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
      sx={{
        width: '100%',
        minWidth: 0,
        pb: 1
      }}
      role="img"
      aria-label={spec.title || 'Bar chart'}
    >
      <BarChart
        width={dimensions.width}
        height={dimensions.height}
        series={chartData.series}
        xAxis={chartData.xAxis}
        yAxis={chartData.yAxis}
        dataset={chartData.dataset}
        layout={spec.layout || 'vertical'}
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
