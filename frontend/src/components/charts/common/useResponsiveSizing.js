import { useMemo } from 'react';

/**
 * Enhanced responsive chart sizing utility
 * Provides breakpoint-aware sizing and optimal chart dimensions
 */
export const useResponsiveChartDimensions = (baseWidth = 500, baseHeight = 320, minWidth = 280) => {
  return useMemo(() => {
    // Calculate optimal dimensions based on width
    const isSmall = baseWidth < 360;
    const isMobile = baseWidth < 480;
    const isTablet = baseWidth < 768;

    const dimensions = {
      width: Math.max(minWidth, baseWidth),
      height: baseHeight,
      isSmall,
      isMobile,
      isTablet,
      isDesktop: !isTablet,
      breakpoint: isSmall ? 'xs' : isMobile ? 'sm' : isTablet ? 'md' : 'lg'
    };

    // Adjust height based on width
    if (isSmall) {
      dimensions.height = Math.min(280, baseHeight);
    } else if (isMobile) {
      dimensions.height = Math.min(300, baseHeight);
    }

    return dimensions;
  }, [baseWidth, baseHeight, minWidth]);
};

/**
 * Hook for responsive font and spacing calculations
 */
export const useResponsiveSizing = (breakpoint = 'md') => {
  return useMemo(() => {
    const sizes = {
      xs: {
        tickFontSize: 10,
        labelFontSize: 9,
        padding: 8,
        gap: 4
      },
      sm: {
        tickFontSize: 11,
        labelFontSize: 10,
        padding: 12,
        gap: 6
      },
      md: {
        tickFontSize: 12,
        labelFontSize: 11,
        padding: 16,
        gap: 8
      },
      lg: {
        tickFontSize: 13,
        labelFontSize: 12,
        padding: 20,
        gap: 10
      }
    };

    return sizes[breakpoint] || sizes.md;
  }, [breakpoint]);
};

/**
 * Hook for margin adjustments based on content
 */
export const useAdaptiveMargins = (config = {}) => {
  return useMemo(() => {
    const {
      hasLongLabels = false,
      hasMultilineTitle = false,
      isVerticalBar = false,
      hasLegend = false
    } = config;

    let margins = {
      top: 24,
      right: 24,
      bottom: 40,
      left: 60
    };

    if (hasLongLabels) {
      if (isVerticalBar) {
        margins.bottom += 20;
      } else {
        margins.left += 20;
      }
    }

    if (hasMultilineTitle) {
      margins.top += 16;
    }

    if (hasLegend) {
      margins.top += 24;
    }

    return margins;
  }, [config]);
};
