import React, { useMemo, useCallback } from 'react';
import { Box, Typography, CircularProgress, Tooltip } from '@mui/material';
import { PieChart } from '@mui/x-charts/PieChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import { useChartConfig, useValueFormatter, useChartSlotProps } from '../common/useChartConfig';
import { useResponsiveChartDimensions } from '../common/useResponsiveSizing';

/**
 * Enhanced PieChart component with improvements:
 * - Better responsive design with compact mode
 * - Enhanced accessibility
 * - Improved legends with better formatting
 * - Custom tooltip support
 * - Value formatting (currency, percent, etc.)
 * - Loading and error states
 * - Donut chart support
 * - Interactive slice selection
 */
export default function EnhancedPieChart({
  spec = {},
  loading = false,
  error = null,
  onItemClick = null,
  onHighlightChange = null
}) {
  const [chartRef, chartWidth] = useResponsiveChartWidth(400, 300);
  const config = useChartConfig(spec);
  const { formatValue } = useValueFormatter(spec.valueFormatter || {});
  const slotProps = useChartSlotProps(config.theme);

  // Calculate dimensions
  const dimensions = useResponsiveChartDimensions(chartWidth, spec.height || 360);
  const isCompact = dimensions.isMobile;

  // Process and validate data
  const chartData = useMemo(() => {
    if (!spec.series || !Array.isArray(spec.series) || spec.series.length === 0) {
      return { valid: false, reason: 'No data available' };
    }

    try {
      const series = spec.series.map(s => {
        if (!Array.isArray(s.data)) {
          return { data: [] };
        }

        // Process data points
        const dataPoints = s.data.map((point, idx) => ({
          id: point.id || point.x || `item-${idx}`,
          value: point.value || point.y || 0,
          label: point.label || point.id || `Item ${idx + 1}`,
          color: point.color || spec.colors?.[idx]
        }));

        // Calculate total for percentage formatting
        const total = dataPoints.reduce((sum, p) => sum + p.value, 0);

        // Enhanced value formatter
        const enhancedDataPoints = dataPoints.map(p => ({
          ...p,
          displayValue: formatValue(p.value),
          displayPercent: total > 0 ? ((p.value / total) * 100).toFixed(1) : 0
        }));

        return {
          data: enhancedDataPoints,
          innerRadius: s.innerRadius !== undefined ? s.innerRadius : (isCompact ? 40 : 60),
          outerRadius: s.outerRadius !== undefined ? s.outerRadius : (isCompact ? 80 : 130),
          paddingAngle: s.paddingAngle !== undefined ? s.paddingAngle : 2,
          cornerRadius: s.cornerRadius !== undefined ? s.cornerRadius : 4,
          highlightScope: s.highlightScope || { faded: 'global', highlighted: 'item' },
          valueFormatter: (item) => {
            if (spec.valueFormatType === 'percent') {
              return `${item.displayPercent}%`;
            }
            return item.displayValue;
          }
        };
      });

      return {
        valid: true,
        series
      };
    } catch (err) {
      console.error('Error processing pie chart data:', err);
      return { valid: false, reason: 'Invalid data format' };
    }
  }, [spec, isCompact, formatValue]);

  // Custom styling
  const chartSx = useMemo(() => ({
    '& .MuiChartsLegend-label': {
      color: config.theme === 'dark' ? '#e5e7eb' : '#374151',
      fontSize: isCompact ? '11px' : '12px',
      fontWeight: 500
    },
    '& .MuiChartsLegend-root': {
      backgroundColor: 'transparent'
    },
    '& .MuiPieArc-root': {
      transition: 'all 0.2s ease-in-out',
      cursor: 'pointer',
      '&:hover': {
        filter: 'brightness(1.1)',
        opacity: 0.95
      }
    },
    '& .MuiChartsAxis-label': {
      fill: config.theme === 'dark' ? '#d1d5db' : '#4b5563',
      fontSize: '12px'
    },
    width: '100%',
    minWidth: 0
  }), [isCompact, config.theme]);

  // Enhanced legend rendering
  const CustomLegend = useCallback(() => {
    if (!chartData.valid || chartData.series.length === 0) return null;

    const dataPoints = chartData.series[0]?.data || [];
    const total = dataPoints.reduce((sum, p) => sum + p.value, 0);

    return (
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: isCompact ? 1 : 1.5,
          mt: 2,
          px: 1,
          color: config.theme === 'dark' ? '#e5e7eb' : '#374151'
        }}
        role="group"
        aria-label="Pie chart legend"
      >
        {dataPoints.map((item, idx) => {
          const percent = total > 0 ? ((item.value / total) * 100).toFixed(1) : 0;
          return (
            <Tooltip
              key={item.id}
              title={`${item.label}: ${item.displayValue} (${percent}%)`}
              arrow
              placement="top"
            >
              <Box
                sx={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 0.75,
                  minWidth: 0,
                  fontSize: isCompact ? '11px' : '12px',
                  padding: '4px 8px',
                  borderRadius: 1,
                  backgroundColor: config.theme === 'dark' 
                    ? 'rgba(229, 231, 235, 0.05)' 
                    : 'rgba(55, 65, 81, 0.05)',
                  cursor: 'default',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: config.theme === 'dark'
                      ? 'rgba(229, 231, 235, 0.1)'
                      : 'rgba(55, 65, 81, 0.1)'
                  }
                }}
              >
                <Box
                  component="span"
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: item.color || '#3b82f6',
                    flex: '0 0 auto'
                  }}
                />
                <Box
                  component="span"
                  sx={{
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}
                >
                  {item.label}
                </Box>
              </Box>
            </Tooltip>
          );
        })}
      </Box>
    );
  }, [chartData, isCompact, config.theme]);

  // Loading state
  if (loading) {
    return (
      <Box
        ref={chartRef}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 320,
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
          minHeight: 320,
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
          minHeight: 320,
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
      sx={{ width: '100%', minWidth: 0 }}
      role="img"
      aria-label={spec.title || 'Pie chart'}
    >
      <PieChart
        width={dimensions.width}
        height={dimensions.height}
        series={chartData.series}
        margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
        slotProps={slotProps}
        sx={chartSx}
        onItemClick={onItemClick}
        onHighlightChange={onHighlightChange}
        skipAnimation={spec.skipAnimation === true}
      />
      <CustomLegend />
    </Box>
  );
}
