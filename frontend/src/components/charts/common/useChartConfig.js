import { useMemo } from 'react';

/**
 * Enhanced hook for chart configuration
 * Provides responsive sizing, accessibility, and animation settings
 */
export const useChartConfig = (spec = {}) => {
  return useMemo(() => {
    const {
      width = 500,
      height = 320,
      margin = {},
      animation = true,
      accessibility = true,
      showLegend = true,
      showGrid = true,
      showTooltip = true,
      theme = 'dark'
    } = spec;

    const defaultMargin = {
      top: 24,
      right: 24,
      bottom: 40,
      left: 60,
      ...margin
    };

    const animationConfig = animation === true ? {
      skipAnimation: false
    } : animation === false ? {
      skipAnimation: true
    } : {
      skipAnimation: false,
      ...animation
    };

    const tooltipConfig = showTooltip !== false ? {
      trigger: 'item'
    } : null;

    const axisStyle = {
      fontSize: 12,
      fill: theme === 'dark' ? '#e5e7eb' : '#374151'
    };

    const gridStyle = {
      stroke: theme === 'dark' ? 'rgba(229, 231, 235, 0.1)' : 'rgba(55, 65, 81, 0.1)'
    };

    return {
      width,
      height,
      margin: defaultMargin,
      animation: animationConfig,
      tooltip: tooltipConfig,
      showLegend,
      showGrid,
      axisStyle,
      gridStyle,
      theme
    };
  }, [spec]);
};

/**
 * Hook for chart margin calculations
 */
export const useChartMargins = (spec = {}) => {
  return useMemo(() => {
    const { hasTitle = false, hasLegend = false, hasSubtitle = false } = spec;
    
    let topMargin = 24;
    if (hasTitle) topMargin += 20;
    if (hasSubtitle) topMargin += 12;
    
    return {
      top: topMargin,
      right: 24,
      bottom: 40 + (hasLegend ? 20 : 0),
      left: 60
    };
  }, [spec]);
};

/**
 * Hook for formatting values based on spec
 */
export const useValueFormatter = (spec = {}) => {
  return useMemo(() => {
    const { format = 'auto', precision = 2 } = spec;

    const formatValue = (value) => {
      if (value === null || value === undefined) return 'N/A';

      if (format === 'currency') {
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: precision,
          maximumFractionDigits: precision
        }).format(value);
      }

      if (format === 'percent') {
        return `${(value * 100).toFixed(precision)}%`;
      }

      if (format === 'number') {
        return new Intl.NumberFormat('en-US', {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision
        }).format(value);
      }

      // Auto format
      if (Math.abs(value) >= 1e6) {
        return `${(value / 1e6).toFixed(precision)}M`;
      }
      if (Math.abs(value) >= 1e3) {
        return `${(value / 1e3).toFixed(precision)}K`;
      }
      return value.toFixed(precision);
    };

    return { formatValue, format, precision };
  }, [spec]);
};

/**
 * Hook for accessibility configurations
 */
export const useA11yConfig = () => {
  return useMemo(() => ({
    ariaLabel: 'Data visualization chart',
    role: 'img',
    slotProps: {
      legend: {
        role: 'group',
        'aria-label': 'Chart legend'
      },
      tooltip: {
        role: 'tooltip',
        'aria-live': 'assertive'
      }
    }
  }), []);
};

/**
 * Hook for chart slot configurations
 */
export const useChartSlots = (customSlots = {}) => {
  return useMemo(() => ({
    ...customSlots
  }), [customSlots]);
};

/**
 * Hook for chart slot props with theme support
 */
export const useChartSlotProps = (theme = 'dark', customProps = {}) => {
  return useMemo(() => {
    const textColor = theme === 'dark' ? '#e5e7eb' : '#374151';
    const backgroundColor = theme === 'dark' ? 'rgba(17, 24, 39, 0.95)' : 'rgba(255, 255, 255, 0.95)';

    return {
      legend: {
        position: { vertical: 'top', horizontal: 'middle' },
        sx: {
          color: textColor,
          fontSize: 12,
          fontWeight: 600,
          backgroundColor,
          borderRadius: 1,
          padding: 1,
          backdropFilter: 'blur(4px)'
        },
        ...customProps.legend
      },
      tooltip: {
        contentStyle: {
          backgroundColor,
          border: `1px solid ${theme === 'dark' ? 'rgba(229, 231, 235, 0.2)' : 'rgba(55, 65, 81, 0.2)'}`,
          borderRadius: 8,
          color: textColor,
          padding: '8px 12px',
          fontSize: 12
        },
        labelStyle: {
          color: textColor,
          fontWeight: 600
        },
        ...customProps.tooltip
      },
      ...customProps
    };
  }, [theme, customProps]);
};
