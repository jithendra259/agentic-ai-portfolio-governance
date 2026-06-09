---
title: "Frontend Charts Enhancement Guide"
description: "Comprehensive guide to improved chart components using MUI X Charts best practices"
lastUpdated: "2024"
---

# Frontend Charts Enhancement Guide

This guide provides a complete overview of the enhanced chart components built using MUI X Charts 9.4.0 best practices.

## Table of Contents

1. [Overview](#overview)
2. [New Components](#new-components)
3. [Migration Guide](#migration-guide)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## Overview

The enhanced chart system provides:

- **Better Responsive Design**: Automatic breakpoint detection and adaptive layouts
- **Enhanced Accessibility**: WCAG AA compliance with proper ARIA labels
- **Improved Tooltips**: Smart formatting with value conversion support
- **Better Error Handling**: Loading states, error messages, no-data fallbacks
- **Theme Integration**: Dark/light mode support with automatic styling
- **Configuration Centralization**: Reusable hooks for consistent behavior
- **Value Formatting**: Currency, percentages, auto-scaling, custom formatters
- **Color Management**: Multiple palettes with WCAG AA compliance checking

---

## New Components

### 1. EnhancedBarChart

**Location**: `/frontend/src/components/charts/bar/EnhancedBarChart.jsx`

**Features**:
- Vertical and horizontal layouts
- Multiple series support
- Responsive sizing with breakpoint awareness
- Value formatting (currency, percent, numbers)
- Grid visualization
- Animation control
- Click and highlight callbacks

**Basic Usage**:

```jsx
import EnhancedBarChart from './EnhancedBarChart';

export function MyComponent() {
  const spec = {
    title: 'Sales by Region',
    series: [
      {
        label: 'Q1',
        data: [120, 132, 101, 134],
        color: '#3b82f6'
      },
      {
        label: 'Q2',
        data: [220, 182, 191, 234],
        color: '#ef4444'
      }
    ],
    xAxis: [
      {
        id: 'x-axis',
        data: ['North', 'South', 'East', 'West'],
        scaleType: 'band'
      }
    ],
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD'
    },
    layout: 'vertical' // or 'horizontal'
  };

  return (
    <EnhancedBarChart 
      spec={spec}
      onItemClick={(event, params) => console.log('Clicked:', params)}
    />
  );
}
```

### 2. EnhancedLineChart

**Location**: `/frontend/src/components/charts/line/EnhancedLineChart.jsx`

**Features**:
- Time-series support (linear, time, point scales)
- Multiple axis support
- Area visualization
- Curve type control (linear, catmullRom, monotoneX, etc.)
- Custom tooltip formatting
- Responsive marker sizing
- Animation with duration control

**Basic Usage**:

```jsx
import EnhancedLineChart from './EnhancedLineChart';

export function TimeSeriesComponent() {
  const spec = {
    title: 'Revenue Trend',
    series: [
      {
        label: 'Revenue',
        data: [
          { date: '2024-01-01', value: 1000 },
          { date: '2024-01-02', value: 1100 },
          { date: '2024-01-03', value: 950 },
          // ...
        ],
        color: '#10b981',
        curve: 'linear',
        area: false,
        showMark: true
      }
    ],
    xAxisType: 'time',
    xLabel: 'Date',
    yLabel: 'Revenue ($)',
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD'
    }
  };

  return <EnhancedLineChart spec={spec} />;
}
```

### 3. EnhancedPieChart

**Location**: `/frontend/src/components/charts/pie/EnhancedPieChart.jsx`

**Features**:
- Pie and donut layouts
- Responsive sizing
- Intelligent legend with tooltips
- Custom value formatting
- Interactive slices
- Percentage display option
- Color management

**Basic Usage**:

```jsx
import EnhancedPieChart from './EnhancedPieChart';

export function CompositionComponent() {
  const spec = {
    title: 'Market Share',
    series: [
      {
        data: [
          { id: 'a', label: 'Product A', value: 400 },
          { id: 'b', label: 'Product B', value: 300 },
          { id: 'c', label: 'Product C', value: 200 },
          { id: 'd', label: 'Product D', value: 100 }
        ],
        innerRadius: 50,
        outerRadius: 130,
        paddingAngle: 2,
        cornerRadius: 4
      }
    ],
    colors: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'],
    valueFormatType: 'percent', // 'percent', 'currency', 'number'
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD'
    }
  };

  return <EnhancedPieChart spec={spec} />;
}
```

---

## New Utility Hooks

### 1. useChartConfig

Centralized configuration management with sensible defaults.

```javascript
const config = useChartConfig(spec);

// Returns:
// {
//   width: number,
//   height: number,
//   theme: 'dark' | 'light',
//   showGrid: boolean,
//   showLegend: boolean,
//   gridStyle: { stroke, dasharray },
//   animation: { duration, easing },
//   accessibility: { role, ariaLabel },
//   tooltipStyle: { ... }
// }
```

### 2. useChartSlotProps

Theme-aware slot properties for legend, tooltip, and axes.

```javascript
const slotProps = useChartSlotProps(theme);

// Usage in BarChart:
<BarChart slotProps={slotProps} />
```

### 3. useValueFormatter

Format values for display (currency, percentages, auto-scaling).

```javascript
const { formatValue } = useValueFormatter({
  type: 'currency',
  locale: 'en-US',
  currency: 'USD'
});

const formatted = formatValue(1234.56); // "$1,234.56"
```

### 4. useResponsiveChartDimensions

Calculate optimal chart dimensions based on container width.

```javascript
const dimensions = useResponsiveChartDimensions(
  containerWidth,
  baseHeight
);

// Returns:
// {
//   width, height,
//   isSmall, isMobile, isTablet, isDesktop,
//   breakpoint: 'xs' | 'sm' | 'md' | 'lg'
// }
```

### 5. useAdaptiveMargins

Dynamic margin calculation based on content.

```javascript
const margins = useAdaptiveMargins({
  hasLongLabels: true,
  hasMultilineTitle: false,
  isVerticalBar: true,
  hasLegend: true
});

// Returns: { top, right, bottom, left }
```

### 6. useResponsiveSizing

Font sizes and spacing that adjust to breakpoints.

```javascript
const sizing = useResponsiveSizing('md');

// Returns:
// {
//   tickFontSize: 12,
//   labelFontSize: 11,
//   padding: 16,
//   gap: 8
// }
```

---

## Color Management

### Color Palettes

**Available palettes**:
- `default`: Professional 8-color palette
- `professional`: Darker, more formal colors
- `pastel`: Light, soft colors
- `categorical`: Distinct, colorful palette
- `diverging`: Blue-to-red range
- `sequential`: Light-to-dark progression
- `heatmap`: Blue-yellow-red spectrum
- `wcagCompliant`: WCAG AA verified colors

### Usage

```javascript
import { getChartColor, COLOR_PALETTES } from './chartColors';

// Get single color
const color = getChartColor(0, 'default'); // '#3b82f6'

// Use entire palette
const colors = COLOR_PALETTES.professional;

// Get theme-aware colors
const themeColors = getThemeColors('dark');
```

---

## Migration Guide

### From SmartBarChartRenderer to EnhancedBarChart

**Before**:
```jsx
<SmartBarChartRenderer 
  spec={{
    series: [...],
    axes: [...]
  }}
/>
```

**After**:
```jsx
<EnhancedBarChart
  spec={{
    title: 'Chart Title',
    series: [{
      label: 'Series 1',
      data: [...]
    }],
    xAxis: [{...}],
    yAxis: [{...}]
  }}
  loading={isLoading}
  error={errorMessage}
  onItemClick={handleClick}
/>
```

### Key Changes

1. **Series Format**: Now an array of objects with `label`, `data`, and optional `color`
2. **Axes Format**: Explicit `xAxis` and `yAxis` arrays
3. **State Management**: Built-in loading and error states
4. **Event Handlers**: `onItemClick` and `onHighlightChange` callbacks
5. **Value Formatting**: Centralized via `valueFormatter` config
6. **Styling**: Automatic theme integration

---

## Configuration

### Spec Object Structure

```typescript
interface ChartSpec {
  // Display
  title?: string;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  
  // Data
  series: Array<{
    label: string;
    data: Array<number | {x/date: any, y/value: any}>;
    color?: string;
    // LineChart specific
    curve?: 'linear' | 'catmullRom' | 'monotoneX' | 'monotoneY';
    area?: boolean;
    showMark?: boolean;
    // PieChart specific
    innerRadius?: number;
    outerRadius?: number;
  }>;
  
  // Axes
  xAxis?: Array<AxisConfig>;
  yAxis?: Array<AxisConfig>;
  
  // Formatting
  valueFormatter?: {
    type: 'currency' | 'percent' | 'number' | 'auto';
    locale?: string;
    currency?: string;
    precision?: number;
  };
  
  // Layout
  layout?: 'vertical' | 'horizontal'; // BarChart only
  xAxisType?: 'point' | 'band' | 'linear' | 'time'; // LineChart only
  
  // Options
  showGrid?: boolean;
  showLegend?: boolean;
  colors?: string[];
  theme?: 'dark' | 'light';
  
  // Performance
  skipAnimation?: boolean;
}
```

---

## Best Practices

### 1. Responsive Design

Always use the responsive hooks:

```jsx
const [ref, width] = useResponsiveChartWidth(500, 300);
const dimensions = useResponsiveChartDimensions(width);

return (
  <Box ref={ref} sx={{ width: '100%' }}>
    <EnhancedBarChart 
      spec={{ ...spec, height: dimensions.height }}
    />
  </Box>
);
```

### 2. Value Formatting

Configure once, use everywhere:

```jsx
const spec = {
  valueFormatter: {
    type: 'currency',
    locale: 'en-US',
    currency: 'USD',
    precision: 2
  }
};
```

### 3. Accessibility

Always provide labels:

```jsx
<EnhancedBarChart
  spec={{
    title: 'Sales Data',
    xLabel: 'Regions',
    yLabel: 'Revenue (USD)',
    series: [...]
  }}
/>
```

### 4. Error Handling

Use loading and error states:

```jsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

return (
  <EnhancedBarChart
    spec={data}
    loading={loading}
    error={error}
  />
);
```

### 5. Performance

For large datasets:

```jsx
// Use skipAnimation for initial render
<EnhancedBarChart
  spec={{ ...spec, skipAnimation: true }}
/>

// Memoize expensive calculations
const spec = useMemo(() => processData(data), [data]);
```

---

## Troubleshooting

### Chart Not Showing

1. Check that `spec.series` has data
2. Verify axis configuration
3. Ensure container has width/height
4. Check browser console for errors

### Values Not Formatting

1. Verify `valueFormatter` configuration
2. Check locale string validity
3. Test with simple number first

### Responsive Issues

1. Ensure parent container has width set
2. Check `useResponsiveChartWidth` is properly used
3. Test at different viewport sizes

### Accessibility Issues

1. Add `title` to spec
2. Include `xLabel` and `yLabel`
3. Use semantic HTML (`role="img"`)
4. Test with screen reader

---

## Performance Optimization

### 1. Data Processing

Memoize expensive operations:

```jsx
const processedSeries = useMemo(() => {
  return spec.series.map(s => ({
    ...s,
    data: s.data.filter(d => d.value > 0)
  }));
}, [spec.series]);
```

### 2. Animation Control

Disable for large datasets:

```jsx
<EnhancedBarChart spec={{ ...spec, skipAnimation: true }} />
```

### 3. Component Memoization

Wrap components for expensive re-renders:

```jsx
export const MemoizedChart = React.memo(EnhancedBarChart);
```

### 4. Data Aggregation

Reduce points for better performance:

```jsx
const aggregateData = (data, bucketSize) => {
  // Group data into buckets
  // Calculate average/sum per bucket
  return aggregated;
};
```

---

## Additional Resources

- [MUI X Charts Documentation](https://mui.com/x/api/charts/)
- [Responsive Design Patterns](https://material.io/design/platform-guidance/android-bars.html)
- [WCAG AA Accessibility Guidelines](https://www.w3.org/WAI/WCAG2AA-Conformance)
- [Color Contrast Checker](https://webaim.org/resources/contrastchecker/)

---

## Version History

- **v1.0.0** (2024): Initial enhanced chart system
  - EnhancedBarChart component
  - EnhancedLineChart component
  - EnhancedPieChart component
  - Utility hooks for configuration
  - Color management system
  - Responsive sizing system
