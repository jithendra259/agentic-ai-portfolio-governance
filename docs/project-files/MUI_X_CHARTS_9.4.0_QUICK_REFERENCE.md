# MUI X Charts v9.4.0 - Quick Reference Guide

## 📊 Chart Components Quick Reference

### 1. BarChart
```typescript
import { BarChart } from '@mui/x-charts/BarChart';

<BarChart
  width={500}
  height={300}
  series={[{ data: [4, 3, 5, 11, 2] }]}
  xAxis={[{ scaleType: 'band', data: ['A', 'B', 'C', 'D', 'E'] }]}
  margin={{ top: 10, bottom: 30, left: 40, right: 10 }}
  slotProps={{
    legend: { hidden: false }
  }}
/>
```

**Key Props**:
- `series`: BarSeries[] - Data series
- `xAxis`: CartesianAxisConfig[] - X-axis configuration
- `yAxis`: CartesianAxisConfig[] - Y-axis configuration
- `layout`: 'vertical' | 'horizontal' - Bar direction
- `grid`: { vertical?, horizontal? } - Grid display
- `margin`: { top?, bottom?, left?, right? } - Margins
- `colors`: string[] - Series colors
- `sx`: SxProps - Custom styling
- `slots`: BarChartSlots - Custom components
- `slotProps`: BarChartSlotProps - Slot props
- `showToolbar`: boolean - Show export toolbar

**Available Slots**:
- `legend` - Legend component
- `tooltip` - Tooltip component
- `axisTickLabelStyle` - Axis labels
- `axisContent` - Axis content

---

### 2. LineChart
```typescript
import { LineChart } from '@mui/x-charts/LineChart';

<LineChart
  width={500}
  height={300}
  series={[
    { data: [1, 4, 2, 5, 7, 2, 4] },
    { data: [3, 6, 2, 5, 4, 1, 7] }
  ]}
  xAxis={[{ scaleType: 'point', data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'] }]}
/>
```

**Key Props**:
- `series`: LineSeriesType[] - Line series
- `xAxis`: CartesianAxisConfig[] - X-axis
- `yAxis`: CartesianAxisConfig[] - Y-axis
- `margin`: MarginConfig - Margins
- `colors`: string[] - Line colors
- `curve`: CurveType - Interpolation type
- `skipAnimation`: boolean - Skip animations
- `slots`: LineChartSlots - Custom components
- `slotProps`: LineChartSlotProps - Slot props

**Curve Types**:
- `'linear'` - Straight lines
- `'natural'` - Natural cubic spline
- `'catmullRom'` - Catmull-Rom curve
- `'monotoneX'` - Monotone X curve
- `'monotoneY'` - Monotone Y curve
- `'step'` - Step function
- `'stepBefore'` - Step (before)
- `'stepAfter'` - Step (after)

---

### 3. PieChart
```typescript
import { PieChart } from '@mui/x-charts/PieChart';

<PieChart
  width={400}
  height={300}
  series={[
    {
      data: [
        { id: 0, value: 10, label: 'A' },
        { id: 1, value: 15, label: 'B' },
        { id: 2, value: 20, label: 'C' }
      ]
    }
  ]}
/>
```

**Key Props**:
- `series`: PieSeriesType[] - Pie series
- `colors`: string[] - Slice colors
- `margin`: MarginConfig - Margins
- `slotProps`: PieChartSlotProps - Slot props

**Series Configuration**:
```typescript
{
  data: Array<{ id, value, label, color? }>;
  innerRadius?: number;     // For donut charts
  outerRadius?: number;
  paddingAngle?: number;
  startAngle?: number;
  endAngle?: number;
  cornerRadius?: number;
  arcLabel?: (params) => string;
  arcLabelMinAngle?: number;
}
```

---

### 4. ScatterChart
```typescript
import { ScatterChart } from '@mui/x-charts/ScatterChart';

<ScatterChart
  width={400}
  height={300}
  series={[
    {
      data: [
        { x: 100, y: 200, id: 1 },
        { x: 120, y: 100, id: 2 }
      ]
    }
  ]}
  xAxis={[{ type: 'linear' }]}
  yAxis={[{ type: 'linear' }]}
/>
```

---

### 5. RadarChart
```typescript
import { RadarChart } from '@mui/x-charts/RadarChart';

<RadarChart
  width={400}
  height={300}
  series={[
    { data: [65, 59, 90, 81, 56] }
  ]}
  xAxis={[{
    data: ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
  }]}
/>
```

---

### 6. Gauge
```typescript
import { Gauge } from '@mui/x-charts/Gauge';

<Gauge
  value={65}
  startAngle={0}
  endAngle={360}
  innerRadius={80}
  outerRadius={120}
/>
```

**Key Props**:
- `value`: number - Current value (0-100 default)
- `valueMin`: number - Minimum value
- `valueMax`: number - Maximum value
- `startAngle`: number - Start angle (degrees)
- `endAngle`: number - End angle (degrees)
- `innerRadius`: number - Inner radius
- `outerRadius`: number - Outer radius
- `colors`: string[] - Color ranges
- `text`: string - Center text

---

### 7. SparkLineChart
```typescript
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';

<SparkLineChart
  data={[1, 4, 3, 5, 1, 6, 4, 2, 5]}
  width={100}
  height={40}
/>
```

**Key Props**:
- `data`: number[] - Values
- `width`: number - Chart width
- `height`: number - Chart height
- `type`: 'line' | 'bar' - Chart type
- `colors`: string[] - Colors
- `curve`: CurveType - Curve type (for lines)

---

## 🎨 Core Shared Components

### ChartsContainer
```typescript
<ChartsContainer
  width={500}
  height={300}
  series={series}
  margin={margin}
>
  {/* Chart content */}
</ChartsContainer>
```

### ChartsAxis
```typescript
<ChartsAxis
  position="left"
  orientation="y"
  scale={yScale}
  tickLabelStyle={tickLabelStyle}
/>
```

### ChartsLegend
```typescript
<ChartsLegend
  position={{ horizontal: 'right', vertical: 'top' }}
  hidden={false}
  slotProps={{
    legend: { direction: 'column' }
  }}
/>
```

### ChartsTooltip
```typescript
<ChartsTooltip
  trigger="item"
  formatter={(params) => params.value}
/>
```

### ChartsGrid
```typescript
<ChartsGrid
  vertical={true}
  horizontal={true}
/>
```

### ChartsLabel
```typescript
<ChartsLabel
  dataKey="label"
  position="top"
/>
```

### ChartsAxisHighlight
```typescript
<ChartsAxisHighlight
  highlight="band"  // 'band' | 'line' | 'none'
/>
```

### ChartsReferenceLine
```typescript
<ChartsReferenceLine
  y={50}
  label="Target"
  lineStyle={{ stroke: 'red', strokeDasharray: '5 5' }}
/>
```

---

## 🎣 Hooks Reference

### Data Access Hooks

```typescript
// Access axis data
const axis = useAxis();

// Get coordinate mapping
const { xScale, yScale } = useScale();

// Get series data
const series = useSeries();
const barSeries = useBarSeries();
const lineSeries = useLineSeries();
const pieSeries = usePieSeries();
const scatterSeries = useScatterSeries();
const radarSeries = useRadarSeries();

// Get drawing area bounds
const { width, height, margin } = useDrawingArea();

// Get color scale
const colorScale = useColorScale();

// Get dataset
const dataset = useDataset();
```

### Interaction Hooks

```typescript
// Get focused item
const focusedItem = useFocusedItem();

// Check if item is focused
const isFocused = useIsItemFocused({ seriesId: 'A', dataIndex: 0 });

// Get highlight state
const isHighlighted = useItemHighlightState({ seriesId: 'A' });

// Get interaction props
const props = useInteractionItemProps({ seriesId: 'A', dataIndex: 0 });

// Get legend state
const { visibleSeries, toggleSeries } = useLegend();
```

### Utility Hooks

```typescript
// Get chart ID
const chartId = useChartId();

// Check if mounted (SSR)
const isMounted = useMounted();

// Get localization strings
const localization = useChartsLocalization();

// Get gradient ID
const gradientId = useChartGradientId();

// Access chart API
const api = useChartApiContext();
```

### Advanced Hooks

```typescript
// Configure axis
const axis = useAxis({
  id: 'x',
  scaleType: 'linear',
  data: data,
  min: 0,
  max: 100
});

// Calculate ticks
const ticks = useTicks({
  scale: scale,
  numTicks: 10
});

// Coordinate transformation
const coordinates = useAxisCoordinates([10, 20], 'x');
```

---

## 🎨 Color Palettes

```typescript
// Available palettes
import {
  blueChartsPalette,
  purpleChartsPalette,
  cyanChartsPalette,
  orangeChartsPalette,
  redChartsPalette,
  greenChartsPalette,
  pinkChartsPalette
} from '@mui/x-charts/colorPalettes';

// Usage in theme
const theme = createTheme({
  palette: {
    mode: 'light'
  }
});

<ChartContainer
  slotProps={{
    legend: {
      colors: blueChartsPalette
    }
  }}
/>
```

---

## 🌍 Localization

```typescript
import { ChartsLocalizationProvider } from '@mui/x-charts';
import { deDE, frFR } from '@mui/x-charts/locales';

<ChartsLocalizationProvider messages={deDE}>
  <BarChart {...props} />
</ChartsLocalizationProvider>
```

**Available Locales**:
- `enUS` - English (US)
- `deDE` - German
- `frFR` - French
- `esES` - Spanish
- `ptBR` - Portuguese (Brazil)
- `zhCN` - Chinese (Simplified)
- `jaJP` - Japanese
- And more...

---

## 📋 Type Definitions Summary

### Series Types
```typescript
// Bar Series
interface BarSeriesType {
  type: 'bar';
  id: string;
  dataKey: string;
  data: number[];
  layout?: 'vertical' | 'horizontal';
  stack?: string;
  color?: string;
  highlightScope?: HighlightScope;
}

// Line Series
interface LineSeriesType {
  type: 'line';
  id: string;
  data: number[];
  curve?: CurveType;
  showMark?: boolean;
  color?: string;
}

// Pie Series
interface PieSeriesType {
  type: 'pie';
  id: string;
  data: Array<{ id: any; value: number; label?: string }>;
  innerRadius?: number;
  outerRadius?: number;
}

// Scatter Series
interface ScatterSeriesType {
  type: 'scatter';
  data: Array<{ x: number; y: number; id?: any }>;
}

// Radar Series
interface RadarSeriesType {
  type: 'radar';
  data: number[];
  label?: string;
}
```

### Axis Configuration
```typescript
interface CartesianAxisConfig {
  id?: string;
  scaleType?: 'linear' | 'log' | 'time' | 'band' | 'point';
  data?: (string | number)[];
  label?: string;
  min?: number;
  max?: number;
  nice?: boolean;
  timezone?: string;
}

interface PolarAxisConfig {
  type?: 'angular' | 'radial';
  data?: (string | number)[];
  min?: number;
  max?: number;
}
```

### Scale Types
```typescript
type ScaleType = 
  | 'linear'     // Linear scale
  | 'log'        // Logarithmic scale
  | 'time'       // Time scale
  | 'band'       // Band scale (categorical)
  | 'point'      // Point scale (categorical)
  | 'utc'        // UTC time scale
  | 'sqrt'       // Square root scale
  | 'power'      // Power scale
  | 'quantile'   // Quantile scale
  | 'quantize'   // Quantize scale
  | 'threshold'  // Threshold scale
```

---

## 🔌 Plugin System

```typescript
// Define custom plugin
const customPlugin: ChartPlugin = {
  seriesProcessor: (series) => {
    // Process series data
    return series;
  },
  transformer: (state, props) => {
    // Transform chart state
    return state;
  }
};

// Use plugin
<BarChart
  plugins={[customPlugin]}
/>
```

---

## 📱 Responsive Usage

```typescript
import { useTheme, useMediaQuery } from '@mui/material';

function ResponsiveChart() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  return (
    <BarChart
      width={isMobile ? 300 : 500}
      height={isMobile ? 200 : 300}
      margin={isMobile ? { top: 5, right: 5, bottom: 5, left: 5 } : { top: 10, right: 30, bottom: 30, left: 60 }}
    />
  );
}
```

---

## 🎯 Best Practices

### 1. **Type Safety**
```typescript
// ✅ Good
const series: BarSeriesType[] = [
  { type: 'bar', id: 'series1', data: [1, 2, 3] }
];

// ❌ Bad
const series = [{ data: [1, 2, 3] }]; // Missing type info
```

### 2. **Props Memoization**
```typescript
// ✅ Good
const chartSeries = useMemo(() => series, [series]);
const chartXAxis = useMemo(() => xAxis, [xAxis]);

// Use memoized props
<BarChart series={chartSeries} xAxis={chartXAxis} />
```

### 3. **Error Handling**
```typescript
// ✅ Good - Validate data before rendering
if (!series || series.length === 0) {
  return <p>No data available</p>;
}

<BarChart series={series} />
```

### 4. **Responsive Design**
```typescript
// ✅ Good - Use ResponsiveChartContainer or hooks
import { ResponsiveChartContainer } from '@mui/x-charts';

<ResponsiveChartContainer>
  <BarChart {...props} />
</ResponsiveChartContainer>
```

### 5. **Accessibility**
```typescript
// ✅ Good - Provide meaningful labels
<BarChart
  xAxis={[{ 
    scaleType: 'band',
    data: ['Jan', 'Feb', 'Mar'],
    label: 'Months' 
  }]}
  yAxis={[{
    label: 'Revenue ($)'
  }]}
  aria-label="Monthly revenue chart"
/>
```

### 6. **Performance**
```typescript
// ✅ Good - Skip animation for large datasets
<BarChart
  series={largeSeries}
  skipAnimation={largeSeries[0].data.length > 1000}
/>

// ✅ Good - Use key for list rendering
{charts.map(chart => (
  <BarChart key={chart.id} {...chart.props} />
))}
```

---

## 🚀 Common Patterns

### Stacked Bar Chart
```typescript
<BarChart
  series={[
    { data: [2, 4, 3], stack: 'A' },
    { data: [4, 5, 4], stack: 'A' },
    { data: [3, 3, 3], stack: 'B' }
  ]}
/>
```

### Multi-Axis Chart
```typescript
<LineChart
  series={[
    { data: [1, 2, 3], yAxisId: 'leftAxis' },
    { data: [100, 200, 150], yAxisId: 'rightAxis' }
  ]}
  yAxis={[
    { id: 'leftAxis', position: 'left' },
    { id: 'rightAxis', position: 'right' }
  ]}
/>
```

### Custom Tooltip
```typescript
<BarChart
  slots={{
    tooltip: CustomTooltip
  }}
  slotProps={{
    tooltip: {
      trigger: 'item',
      contentStyle: { color: 'red' }
    }
  }}
/>
```

### Date Axis
```typescript
<LineChart
  xAxis={[{
    scaleType: 'time',
    data: [new Date(2024, 0, 1), new Date(2024, 0, 2)]
  }]}
/>
```

### Logarithmic Scale
```typescript
<LineChart
  yAxis={[{
    scaleType: 'log',
    min: 1,
    max: 10000
  }]}
/>
```

---

## 📚 API Documentation Links

- **Main Docs**: https://mui.com/x/react-charts/
- **Bar Charts**: https://mui.com/x/react-charts/bars/
- **Line Charts**: https://mui.com/x/react-charts/lines/
- **Pie Charts**: https://mui.com/x/react-charts/pie/
- **Scatter Charts**: https://mui.com/x/react-charts/scatter/
- **Radar Charts**: https://mui.com/x/react-charts/radar/
- **Gauge**: https://mui.com/x/react-charts/gauge/
- **API Reference**: https://mui.com/x/api/charts/

---

**Quick Reference Generated**: 2026-06-08  
**MUI X Version**: 9.4.0
