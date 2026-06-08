import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import { PieChart } from '@mui/x-charts/PieChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';
import { ScatterChartPro } from '@mui/x-charts-pro/ScatterChartPro';
import { ScatterChartPremium } from '@mui/x-charts-premium/ScatterChartPremium';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { Heatmap } from '@mui/x-charts-premium/Heatmap';
import { SankeyChart } from '@mui/x-charts-premium/SankeyChart';
import { FunnelChart } from '@mui/x-charts-premium/FunnelChart';
import { RadarChart } from '@mui/x-charts-premium/RadarChart';
import { Gauge } from '@mui/x-charts-premium/Gauge';
import { Unstable_RadialBarChart as RadialBarChart } from '@mui/x-charts-premium/RadialBarChart';
import { Unstable_RadialLineChart as RadialLineChart } from '@mui/x-charts-premium/RadialLineChart';
import { Box, Typography, Paper, Chip, Skeleton } from '@mui/material';
import { useDrawingArea, useXScale, useYScale, useAnimateBarLabel } from '@mui/x-charts/hooks';
import { ChartsClipPath } from '@mui/x-charts/ChartsClipPath';
import { useTheme, alpha } from '@mui/material/styles';
import { BACKEND_BASE } from '../config/api';
import SmartBarChartRenderer from './charts/bar/SmartBarChartRenderer';
import {
  CandlestickChart,
  FunnelChartRenderer,
  GaugeChartRenderer,
  HeatmapChartRenderer,
  LineChartRenderer,
  NetworkChartRenderer,
  PieChartRenderer,
  RadarChartRenderer,
  RadialBarChartRenderer,
  RadialLineChartRenderer,
  SankeyChartRenderer,
  ScatterChartRenderer,
  SparklineChartRenderer,
} from './charts';

// Must match PALETTE in generate_dynamic_plot.py
const PALETTE = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#f97316',
];

const CHART_HEIGHT = 320;

const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };
const GRID_STYLE = { stroke: '#374151', strokeDasharray: '4 4' };

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getResponsiveChartHeight(spec, fallback = CHART_HEIGHT) {
  const requested = toFiniteNumber(spec?.height, fallback);
  return Math.max(180, Math.min(requested, 720));
}

function getPieShare(value, total) {
  if (!total) return 0;
  return (value / total) * 100;
}

function useResponsiveChartWidth(fallback = 360) {
  const ref = React.useRef(null);
  const [width, setWidth] = React.useState(fallback);

  React.useEffect(() => {
    if (!ref.current) return;
    const updateWidth = (value) => {
      setWidth(Math.max(280, Math.floor(value || fallback)));
    };
    updateWidth(ref.current.getBoundingClientRect().width);
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) {
        updateWidth(entries[0].contentRect.width);
      }
    });
    resizeObserver.observe(ref.current);
    return () => resizeObserver.disconnect();
  }, [fallback]);

  return [ref, width];
}

function getValueFormatter(format) {
  if (format === 'percent' || format === '%') {
    return v => v == null ? '' : `${v.toFixed(1)}%`;
  }
  if (format === 'decimal') {
    return v => v == null ? '' : Number(v).toFixed(2);
  }
  if (format === 'beta') {
    return v => v == null ? '' : `${Number(v).toFixed(2)} beta`;
  }
  if (format === 'k') {
    return v => v == null ? '' : `${(v / 1000).toFixed(1)}k`;
  }
  if (format === 'currency') {
    return v => v == null ? '' : `$${v.toLocaleString()}`;
  }
  return v => {
    if (v == null) return '';
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return v.toFixed(2);
  };
}

function getAnimationSx(animation) {
  if (!animation) return {};
  const duration = animation.duration || '1s';
  const delay = animation.delay || '0s';
  const easing = animation.easing || 'ease-out';

  return {
    '& .MuiLineElement-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
    '& .MuiAreaElement-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
    '& .MuiBarElement-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
    '& .MuiBarLabel-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
    '& .MuiPieArc-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
    '& .MuiPieArcLabel-root.MuiCharts-animate': {
      animationDuration: duration,
      animationDelay: delay,
      animationTimingFunction: easing,
    },
  };
}

function AnimatedBarLabel(props) {
  const {
    seriesId,
    dataIndex,
    color,
    isFaded,
    isHighlighted,
    classes,
    xOrigin,
    yOrigin,
    x,
    y,
    width,
    height,
    layout,
    skipAnimation,
    ...otherProps
  } = props;

  const animatedProps = useAnimateBarLabel({
    xOrigin,
    x,
    yOrigin,
    y,
    width,
    height,
    layout,
    skipAnimation,
  });

  return (
    <text
      {...otherProps}
      fill={color || otherProps.fill || '#e5e7eb'}
      textAnchor="middle"
      dominantBaseline="central"
      {...animatedProps}
      style={{ fontSize: 11, fontWeight: 500 }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RecessionBands component
// ─────────────────────────────────────────────────────────────────────────────
function RecessionBands({ periods }) {
  const { top, left, width, height } = useDrawingArea();
  const xScale = useXScale();
  const theme = useTheme();
  const labelFill = alpha(theme.palette.text.primary, 0.7);

  if (!periods || !Array.isArray(periods)) return null;

  return (
    <g>
      {periods.map((p, index) => {
        if (!p.start || !p.end) return null;
        const startDate = new Date(p.start);
        const endDate = new Date(p.end);

        const xStart = xScale(startDate);
        const xEnd = xScale(endDate);

        if (xStart === undefined || xEnd === undefined || isNaN(xStart) || isNaN(xEnd)) {
          return null;
        }

        let startX = xStart;
        let endX = xEnd;

        // Clip to drawing area bounds
        if (startX < left) startX = left;
        if (endX > left + width) endX = left + width;

        if (startX >= endX) return null;

        const textX = xStart >= left ? xStart : left;

        return (
          <React.Fragment key={index}>
            <rect
              x={startX}
              y={top}
              width={endX - startX}
              height={height}
              fill="grey"
              opacity={0.15}
            />
            <text
              x={textX + 4}
              y={top - 5}
              textAnchor="start"
              dominantBaseline="auto"
              fill={labelFill}
              fontSize="0.75rem"
              fontWeight={500}
              pointerEvents="none"
            >
              {p.label}
            </text>
          </React.Fragment>
        );
      })}
    </g>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Line chart — spec.series[].data = [{x: "YYYY-MM-DD", y: number}]
// ─────────────────────────────────────────────────────────────────────────────

function SpecLineChart({ spec }) {
  const { dataset, series, yAxisConfig, margins } = useMemo(() => {
    if (!spec?.series?.length) return { dataset: [], series: [], yAxisConfig: [], margins: { top: 24, right: 24, left: 60, bottom: 40 } };

    const dateSet = new Set();
    spec.series.forEach(s => s.data?.forEach(pt => dateSet.add(pt.x)));
    const sortedDates = Array.from(dateSet).sort();

    const byDate = {};
    sortedDates.forEach(d => { byDate[d] = { date: new Date(d) }; });
    spec.series.forEach(s => {
      s.data?.forEach(pt => {
        if (byDate[pt.x]) byDate[pt.x][s.name] = pt.y;
      });
    });

    const dataset = sortedDates.map(d => byDate[d]);

    // ── Build series config with all MUI X line chart features ──
    const series = spec.series.map((s, i) => {
      const entry = {
        type: 'line',
        dataKey: s.name,
        label: s.label || s.name,
        color: s.color || PALETTE[i % PALETTE.length],
        valueFormatter: getValueFormatter(s.value_format || spec.y_format),
      };

      // ── Y-axis binding ──
      if (s.yAxisId) entry.yAxisId = s.yAxisId;

      // ── Area fill ──
      if (s.area != null) entry.area = s.area;

      // ── Area baseline ──
      if (s.baseline != null) entry.baseline = s.baseline;

      // ── Stacking ──
      if (s.stack) entry.stack = s.stack;
      if (s.stackOffset) entry.stackOffset = s.stackOffset;

      // ── Curve interpolation (per-series overrides global) ──
      if (s.curve) {
        entry.curve = s.curve;
      } else if (spec.curve) {
        entry.curve = spec.curve;
      }

      // ── Marks ──
      entry.showMark = s.showMark ?? false;
      if (s.shape) entry.shape = s.shape;

      // ── Connect nulls ──
      entry.connectNulls = s.connectNulls ?? spec.connect_nulls ?? spec.connectNulls ?? false;

      // ── Highlight scope (per-series overrides global) ──
      if (s.highlightScope) {
        entry.highlightScope = s.highlightScope;
      } else if (spec.highlightScope) {
        entry.highlightScope = spec.highlightScope;
      }

      // ── Disable highlight ──
      if (s.disableHighlight != null) entry.disableHighlight = s.disableHighlight;

      return entry;
    });

    // ── Y-axis configuration ──
    let yAxisConfig = [];
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      yAxisConfig = spec.yAxis.map(axis => {
        const ax = {
          id: axis.id,
          label: axis.label || '',
          position: axis.position || 'left',
          tickLabelStyle: AXIS_STYLE,
          valueFormatter: getValueFormatter(axis.value_format),
          width: axis.width || (axis.position === 'right' ? 50 : 55),
          domainLimit: axis.domainLimit || 'nice',
        };
        // ── Y-axis colorMap ──
        if (axis.colorMap) ax.colorMap = axis.colorMap;
        return ax;
      });
    } else {
      yAxisConfig = [{
        id: 'default-y-axis',
        tickLabelStyle: AXIS_STYLE,
        label: spec.y_label || '',
        valueFormatter: getValueFormatter(spec.y_format),
        domainLimit: 'nice',
      }];
    }

    const hasRightAxis = spec.yAxis?.some(axis => axis.position === 'right');
    const margins = {
      top: 60,
      right: hasRightAxis ? 60 : 24,
      left: 60,
      bottom: 60
    };

    return { dataset, series, yAxisConfig, margins };
  }, [spec]);

  if (!dataset.length) return null;

  // ── Grid config from backend ──
  const gridConfig = spec.grid || { horizontal: true };

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);
  const [chartRef, chartWidth] = useResponsiveChartWidth();

  return (
    <Box ref={chartRef} sx={{ width: '100%', height: CHART_HEIGHT, minWidth: 0, ...animationSx }}>
      <LineChart
        width={chartWidth}
        dataset={dataset}
        series={series}
        xAxis={[{
          id: 'x-axis',
          dataKey: 'date',
          scaleType: 'time',
          tickLabelStyle: AXIS_STYLE,
          label: spec.x_label || 'Date',
          valueFormatter: (d, context) => {
            if (!d) return '';
            if (context?.location !== 'tick') {
              return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
            }
            return d.getMonth() === 0
              ? d.toLocaleDateString('en-US', { year: 'numeric' })
              : d.toLocaleDateString('en-US', { month: 'short' });
          }
        }]}
        yAxis={yAxisConfig}
        margin={margins}
        grid={gridConfig}
        {...(spec.skipAnimation ? { skipAnimation: true } : {})}
        {...(spec.experimentalFeatures ? { experimentalFeatures: spec.experimentalFeatures } : {})}
        slotProps={{
          legend: {
            position: { vertical: 'top', horizontal: 'middle' },
            padding: 0,
            sx: {
              color: '#e5e7eb',
              fontSize: 12,
              mt: 0,
            },
          }
        }}
        sx={{
          '& .MuiChartsAxis-bottom .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-left .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-right .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiLineElement-root': { strokeWidth: 2 },
          '& .MuiMarkElement-root': { strokeWidth: 2 },
          '& .MuiChartsGrid-line': GRID_STYLE,
          ...animationSx,
        }}
      >
        {spec.recessions && <RecessionBands periods={spec.recessions} />}
      </LineChart>
    </Box>
  );
}

function SpecBarChart({ spec }) {
  const { dataset, series, xAxisConfig, yAxisConfig, chartHeight, margin, categories } = useMemo(() => {
    if (!spec?.series?.length) {
      return { dataset: [], series: [], xAxisConfig: [], yAxisConfig: [], chartHeight: CHART_HEIGHT, margin: {}, categories: [] };
    }
    const isHorizontal = spec.layout === 'horizontal';

    // Collect all unique categories across all series
    const catSet = new Set();
    spec.series.forEach(s => s.data?.forEach(pt => catSet.add(pt.x)));
    const categories = Array.from(catSet);

    // Build pivoted dataset [{label: "category", Score: 10, ...}]
    const byCategory = {};
    categories.forEach(c => { byCategory[c] = { label: c }; });
    spec.series.forEach(s => {
      s.data?.forEach(pt => {
        if (byCategory[pt.x]) byCategory[pt.x][s.name] = toFiniteNumber(pt.y);
      });
    });

    const dataset = categories.map(c => byCategory[c]);
    const longestCategory = categories.reduce((max, label) => Math.max(max, String(label).length), 0);
    const categoryAxisWidth = Math.max(72, Math.min(180, longestCategory * 7 + 28));
    const chartHeight = Math.max(CHART_HEIGHT, categories.length * (isHorizontal ? 34 : 22) + 96);

    // ── Build series config with all MUI X bar chart features ──
    const series = spec.series.map((s, i) => {
      const entry = {
        dataKey: s.name,
        label: s.label || s.name,
        color: s.color || PALETTE[i % PALETTE.length],
        valueFormatter: getValueFormatter(s.value_format || spec.y_format || 'none'),
      };

      // ── Stacking ──
      if (s.stack) entry.stack = s.stack;
      if (s.stackOffset) entry.stackOffset = s.stackOffset;

      // ── Bar labels ──
      if (s.barLabel != null) entry.barLabel = s.barLabel;
      if (s.barLabelPlacement) entry.barLabelPlacement = s.barLabelPlacement;

      // ── Min bar size ──
      if (s.minBarSize != null) entry.minBarSize = s.minBarSize;

      // ── Highlight scope (per-series overrides global) ──
      if (s.highlightScope) {
        entry.highlightScope = s.highlightScope;
      } else if (spec.highlightScope) {
        entry.highlightScope = spec.highlightScope;
      }

      return entry;
    });

    // ── X-axis configuration ──
    let xAxisConfig;
    let yAxisConfig;
    if (isHorizontal) {
      xAxisConfig = spec.xAxis && Array.isArray(spec.xAxis)
        ? spec.xAxis.map(ax => ({
            ...ax,
            tickLabelStyle: AXIS_STYLE,
            valueFormatter: ax.valueFormatter || getValueFormatter(ax.value_format || spec.y_format),
            domainLimit: ax.domainLimit || 'nice',
          }))
        : [{
            tickLabelStyle: AXIS_STYLE,
            label: spec.y_label || 'Value',
            valueFormatter: getValueFormatter(spec.y_format),
            domainLimit: 'nice',
          }];

      yAxisConfig = spec.yAxis && Array.isArray(spec.yAxis)
        ? spec.yAxis.map(ax => ({
            ...ax,
            data: ax.data || categories,
            dataKey: ax.dataKey || 'label',
            scaleType: ax.scaleType || 'band',
            tickLabelStyle: AXIS_STYLE,
            width: ax.width || categoryAxisWidth,
            valueFormatter: ax.valueFormatter || ((value) => String(value ?? '')),
          }))
        : [{
            dataKey: 'label',
            data: categories,
            scaleType: 'band',
            tickLabelStyle: AXIS_STYLE,
            width: categoryAxisWidth,
            label: spec.x_label || 'Category',
            valueFormatter: (value) => String(value ?? ''),
          }];
    } else if (spec.xAxis && Array.isArray(spec.xAxis)) {
      xAxisConfig = spec.xAxis.map(ax => {
        const xAx = {
          ...ax,
          data: ax.data || categories,
          dataKey: ax.dataKey || 'label',
          scaleType: ax.scaleType || 'band',
          tickLabelStyle: { ...AXIS_STYLE, angle: -30, textAnchor: 'end' },
          valueFormatter: ax.valueFormatter || ((value) => String(value ?? '')),
        };
        if (spec.categoryGapRatio != null) xAx.categoryGapRatio = spec.categoryGapRatio;
        if (spec.barGapRatio != null) xAx.barGapRatio = spec.barGapRatio;
        return xAx;
      });
    } else {
      const xAx = {
        dataKey: 'label',
        data: categories,
        scaleType: 'band',
        tickLabelStyle: { ...AXIS_STYLE, angle: -30, textAnchor: 'end' },
        label: spec.x_label || '',
        valueFormatter: (value) => String(value ?? ''),
      };
      if (spec.categoryGapRatio != null) xAx.categoryGapRatio = spec.categoryGapRatio;
      if (spec.barGapRatio != null) xAx.barGapRatio = spec.barGapRatio;
      xAxisConfig = [xAx];
    }

    // ── Y-axis configuration ──
    if (!isHorizontal && spec.yAxis && Array.isArray(spec.yAxis)) {
      yAxisConfig = spec.yAxis.map(ax => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(ax.value_format || spec.y_format),
        domainLimit: ax.domainLimit || 'nice',
      }));
    } else if (!isHorizontal) {
      yAxisConfig = [{
        tickLabelStyle: AXIS_STYLE,
        label: spec.y_label || '',
        valueFormatter: getValueFormatter(spec.y_format),
        domainLimit: 'nice',
      }];
    }

    const margin = isHorizontal
      ? { top: 24, right: 28, left: categoryAxisWidth + 12, bottom: 44 }
      : { top: 24, right: 24, left: 60, bottom: 64 };

    return { dataset, series, xAxisConfig, yAxisConfig, chartHeight, margin, categories };
  }, [spec]);

  if (!dataset.length) return null;

  // ── Grid config from backend ──
  const gridConfig = spec.grid || { horizontal: true };

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);
  const slotsConfig = useMemo(() => {
    if (spec.animation?.animatedLabels !== false) {
      return { barLabel: AnimatedBarLabel };
    }
    return undefined;
  }, [spec.animation]);
  const [chartRef, chartWidth] = useResponsiveChartWidth();

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <BarChart
        width={chartWidth}
        dataset={dataset}
        xAxis={xAxisConfig}
        yAxis={yAxisConfig}
        series={series}
        height={chartHeight}
        margin={margin}
        grid={gridConfig}
        borderRadius={spec.borderRadius ?? 4}
        {...(slotsConfig ? { slots: slotsConfig } : {})}
        {...(spec.layout ? { layout: spec.layout } : {})}
        {...(spec.skipAnimation ? { skipAnimation: true } : {})}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsGrid-line': GRID_STYLE,
          '& .MuiChartsLegend-root': {
            color: '#e5e7eb',
            fontSize: 12,
          },
          ...animationSx,
        }}
        slotProps={{ legend: { sx: { color: '#e5e7eb', fontSize: 12 } } }}
      />
      {spec.layout !== 'horizontal' && categories.length > 0 && (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: `repeat(${categories.length}, minmax(0, 1fr))`,
            pl: `${margin.left}px`,
            pr: `${margin.right}px`,
            mt: -5,
            color: '#9ca3af',
            fontSize: 11,
            lineHeight: 1.2,
          }}
        >
          {categories.map((category) => (
            <Box key={category} sx={{ textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {category}
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Pie Center Label
// ─────────────────────────────────────────────────────────────────────────────
function PieCenterLabel({ children }) {
  const { width, height, left, top } = useDrawingArea();
  const lines = String(children ?? '').split('\n').filter(Boolean);
  const fontSize = lines.length > 2 ? 12 : 16;
  const lineHeight = fontSize + 3;
  const firstDy = lines.length > 1 ? -((lines.length - 1) * lineHeight) / 2 : 0;
  return (
    <text
      x={left + width / 2}
      y={top + height / 2}
      textAnchor="middle"
      dominantBaseline="central"
      fill="#ffffff"
      style={{ fontSize, fontWeight: 'bold' }}
    >
      {lines.map((line, index) => (
        <tspan key={`${line}-${index}`} x={left + width / 2} dy={index === 0 ? firstDy : lineHeight}>
          {line}
        </tspan>
      ))}
    </text>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pie chart — spec.series[0].data = [{x: label, y: value, color?: string}]
// ─────────────────────────────────────────────────────────────────────────────
function SpecPieChart({ spec }) {
  const mappedSeries = useMemo(() => {
    if (!spec?.series || !Array.isArray(spec.series)) return [];

    return spec.series.map((s, sIndex) => {
      const rawData = s.data || [];

      // Calculate series-specific stats for auto formatting
      const total = rawData.reduce((sum, pt) => {
        const val = pt.value != null ? pt.value : (pt.y || 0);
        return sum + val;
      }, 0);
      const maxVal = rawData.reduce((max, pt) => {
        const val = pt.value != null ? pt.value : (pt.y || 0);
        return val > max ? val : max;
      }, 0);
      const isFractional = maxVal <= 1.0;

      // Standardize and sort data points
      let dataPoints = rawData.map((pt, i) => {
        const id = pt.id || pt.x || `slice-${i}`;
        const value = pt.value != null ? pt.value : (pt.y != null ? pt.y : 0);
        const color = pt.color || PALETTE[i % PALETTE.length];

        const entry = { id, value, color };
        if (pt.labelMarkType) entry.labelMarkType = pt.labelMarkType;

        // Format value for legend label
        const formatVal = (v) => {
          if (s.valueFormatter === 'percent' || (!s.valueFormatter && isFractional)) {
            const multiplier = isFractional ? 100 : 1;
            return `${(v * multiplier).toFixed(1)}%`;
          }
          if (s.valueFormatter === 'currency') {
            return `$${v.toLocaleString()}`;
          }
          if (s.valueFormatter === 'raw') {
            return v.toString();
          }
          return `${getPieShare(v, total).toFixed(1)}%`;
        };

        // Legend label: use pt.label if supplied, else build from id/x and value
        entry.label = pt.label != null ? pt.label : `${id} (${formatVal(value)})`;

        return entry;
      });

      // Sorting data values if sorting is specified
      if (s.sorting === 'asc') {
        dataPoints = [...dataPoints].sort((a, b) => a.value - b.value);
      } else if (s.sorting === 'desc') {
        dataPoints = [...dataPoints].sort((a, b) => b.value - a.value);
      }

      const entry = {
        data: dataPoints,
        innerRadius: s.innerRadius !== undefined ? s.innerRadius : 48,
        outerRadius: s.outerRadius !== undefined ? s.outerRadius : 110,
        paddingAngle: s.paddingAngle !== undefined ? s.paddingAngle : 2,
        cornerRadius: s.cornerRadius !== undefined ? s.cornerRadius : 4,
      };

      // Sizing & arc angles props
      if (s.startAngle !== undefined) entry.startAngle = s.startAngle;
      if (s.endAngle !== undefined) entry.endAngle = s.endAngle;
      if (s.cx !== undefined) entry.cx = s.cx;
      if (s.cy !== undefined) entry.cy = s.cy;
      if (s.arcLabelRadius !== undefined) entry.arcLabelRadius = s.arcLabelRadius;
      if (s.arcLabelMinAngle !== undefined) entry.arcLabelMinAngle = s.arcLabelMinAngle;

      // Slice labels (arcLabel): string or function
      if (s.arcLabel) {
        if (typeof s.arcLabel === 'string') {
          if (s.arcLabel === 'percent') {
            entry.arcLabel = (item) => {
              const pct = total > 0 ? (item.value / total) * 100 : 0;
              return `${pct.toFixed(0)}%`;
            };
          } else if (s.arcLabel === 'label-percent') {
            entry.arcLabel = (item) => {
              const pct = total > 0 ? (item.value / total) * 100 : 0;
              const ptLabel = dataPoints.find(d => d.id === item.id)?.label || item.id;
              const cleanLabel = typeof ptLabel === 'string' && ptLabel.includes(' (') ? ptLabel.split(' (')[0] : ptLabel;
              return `${cleanLabel} (${pct.toFixed(0)}%)`;
            };
          } else {
            entry.arcLabel = s.arcLabel;
          }
        } else {
          entry.arcLabel = s.arcLabel;
        }
      }

      // Custom tooltip valueFormatter
      entry.valueFormatter = (item) => {
        const val = item.value;
        if (s.valueFormatter === 'percent' || (!s.valueFormatter && isFractional)) {
          const multiplier = isFractional ? 100 : 1;
          return `${(val * multiplier).toFixed(1)}%`;
        }
        if (s.valueFormatter === 'currency') {
          return `$${val.toLocaleString()}`;
        }
        if (s.valueFormatter === 'raw') {
          return val.toString();
        }
        return `${getPieShare(val, total).toFixed(1)}%`;
      };

      // Highlight options
      if (s.highlightScope) {
        entry.highlightScope = s.highlightScope;
      } else if (spec.highlightScope) {
        entry.highlightScope = spec.highlightScope;
      }
      if (s.faded) entry.faded = s.faded;
      if (s.highlighted) entry.highlighted = s.highlighted;

      return entry;
    });
  }, [spec]);

  const [chartRef, chartWidth] = useResponsiveChartWidth();
  const compact = chartWidth < 460;
  const chartHeight = compact ? Math.min(320, spec.height || 320) : (spec.height || CHART_HEIGHT);
  const centerLabel = spec.centerLabel || spec.center_label;
  const responsiveSeries = useMemo(() => (
    compact
      ? mappedSeries.map((series) => ({
          ...series,
          innerRadius: Math.min(series.innerRadius ?? 0, 48),
          outerRadius: Math.min(series.outerRadius ?? 110, 86),
        }))
      : mappedSeries
  ), [compact, mappedSeries]);

  if (!mappedSeries.length) return null;

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);

  const chartProps = {};
  if (spec.skipAnimation) chartProps.skipAnimation = true;
  chartProps.hideLegend = true;
  if (spec.colors && Array.isArray(spec.colors)) chartProps.colors = spec.colors;
  const legendItems = spec.hideLegend ? [] : responsiveSeries.flatMap((series) => series.data || []);

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <PieChart
        width={chartWidth}
        series={responsiveSeries}
        height={chartHeight}
        margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
        sx={{
          '& .MuiPieArcLabel-root': {
            fontSize: compact ? '10px' : '11px',
            fill: '#ffffff',
            fontWeight: 'bold',
          },
          ...animationSx,
        }}
        slotProps={{
          legend: {
            sx: {
              color: '#e5e7eb',
              fontSize: 11,
              gap: 1.25,
              '& .MuiChartsLegend-mark': {
                width: 10,
                height: 10,
              },
              '& .MuiChartsLegend-series': {
                gap: 0.75,
              },
            },
          },
        }}
        {...chartProps}
      >
        {centerLabel && <PieCenterLabel>{centerLabel}</PieCenterLabel>}
      </PieChart>
      {legendItems.length > 0 && (
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 1.25,
            mt: -1,
            px: 1,
            color: '#e5e7eb',
            fontSize: 11,
            lineHeight: 1.2,
          }}
        >
          {legendItems.map((item) => (
            <Box key={item.id} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
              <Box component="span" sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: item.color, flex: '0 0 auto' }} />
              <Box component="span" sx={{ whiteSpace: 'nowrap' }}>{item.label || item.id}</Box>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Scatter chart
// ─────────────────────────────────────────────────────────────────────────────
function SpecScatterChart({ spec }) {
  const series = useMemo(() => {
    if (!spec?.series || !Array.isArray(spec.series)) return [];
    return spec.series.map((s, i) => {
      const dataPoints = (s.data || []).map((pt, j) => ({
        x: toFiniteNumber(pt.x),
        y: toFiniteNumber(pt.y),
        id: pt.id !== undefined ? pt.id : `pt-${j}`,
        label: pt.label,
        ...(pt.z !== undefined ? { z: toFiniteNumber(pt.z) } : {}),
        ...(pt.sizeValue !== undefined ? { sizeValue: toFiniteNumber(pt.sizeValue) } : {}),
        ...(pt.colorValue !== undefined ? { colorValue: pt.colorValue } : {}),
      }));
      const entry = {
        id: s.id || s.name || `scatter-${i}`,
        type: 'scatter',
        label: s.label || s.name || `Series ${i + 1}`,
        data: dataPoints,
        color: s.color || PALETTE[i % PALETTE.length],
      };
      if (s.markerSize !== undefined) entry.markerSize = s.markerSize;
      if (s.sizeAxisId) entry.sizeAxisId = s.sizeAxisId;
      if (s.colorAxisId) entry.colorAxisId = s.colorAxisId;
      if (s.labelMarkType) entry.labelMarkType = s.labelMarkType;
      
      if (s.highlightScope) {
        entry.highlightScope = s.highlightScope;
      } else if (spec.highlightScope) {
        entry.highlightScope = spec.highlightScope;
      }
      return entry;
    });
  }, [spec]);

  const xAxisConfig = useMemo(() => {
    if (spec.xAxis && Array.isArray(spec.xAxis)) {
      return spec.xAxis.map(ax => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(ax.value_format || spec.x_format || spec.x_unit),
        label: ax.label || spec.x_label || spec.x_axis || '',
        domainLimit: ax.domainLimit || 'nice',
      }));
    }
    return [{
      tickLabelStyle: AXIS_STYLE,
      label: spec.x_label || '',
      valueFormatter: getValueFormatter(spec.x_format || spec.x_unit),
      domainLimit: 'nice',
    }];
  }, [spec]);

  const yAxisConfig = useMemo(() => {
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      return spec.yAxis.map(ax => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(ax.value_format || spec.y_format || spec.y_unit),
        label: ax.label || spec.y_label || spec.y_axis || '',
        domainLimit: ax.domainLimit || 'nice',
      }));
    }
    return [{
      tickLabelStyle: AXIS_STYLE,
      label: spec.y_label || '',
      valueFormatter: getValueFormatter(spec.y_format || spec.y_unit),
      domainLimit: 'nice',
    }];
  }, [spec]);

  const zAxisConfig = useMemo(() => {
    if (spec.zAxis && Array.isArray(spec.zAxis)) {
      return spec.zAxis.map(ax => ({
        ...ax,
      }));
    }
    return undefined;
  }, [spec]);

  const chartProps = {};
  if (spec.skipAnimation) chartProps.skipAnimation = true;
  if (spec.hideLegend) chartProps.hideLegend = true;
  if (spec.colors && Array.isArray(spec.colors)) chartProps.colors = spec.colors;
  if (spec.renderer) chartProps.renderer = spec.renderer;
  chartProps.hitAreaRadius = spec.hitAreaRadius !== undefined ? spec.hitAreaRadius : 20;

  const gridConfig = spec.grid || { horizontal: true, vertical: true };
  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);
  const [chartRef, chartWidth] = useResponsiveChartWidth();
  const chartHeight = getResponsiveChartHeight(spec, 420);
  const ChartComponent = spec.component === 'ScatterChartPremium' || spec.chart_type === 'webgl_scatter' || spec.renderer === 'webgl'
    ? ScatterChartPremium
    : spec.component === 'ScatterChartPro' || spec.chart_type === 'bubble_scatter'
      ? ScatterChartPro
      : ScatterChart;

  if (!series.length) return null;

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0, ...animationSx }}>
      <ChartComponent
        width={chartWidth}
        series={series}
        xAxis={xAxisConfig}
        yAxis={yAxisConfig}
        {...(zAxisConfig ? { zAxis: zAxisConfig } : {})}
        height={chartHeight}
        margin={spec.margin || { top: 46, right: 28, left: 72, bottom: 68 }}
        grid={gridConfig}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-label': { fill: '#e5e7eb', fontSize: 12, fontWeight: 600 },
          '& .MuiChartsGrid-line': GRID_STYLE,
          '& .MuiChartsLegend-root': {
            color: '#e5e7eb',
            fontSize: 12,
          },
          '& .MuiScatter-root .MuiScatter-mark': {
            fillOpacity: spec.chart_type === 'bubble_scatter' ? 0.68 : 0.9,
            strokeWidth: 1.2,
          },
          ...animationSx,
        }}
        slotProps={{
          legend: {
            position: { vertical: 'top', horizontal: 'middle' },
            sx: { color: '#e5e7eb', fontSize: 12 },
          },
        }}
        {...chartProps}
      >
        {spec.regression_line && <ScatterRegressionLine regression={spec.regression_line} />}
      </ChartComponent>
    </Box>
  );
}

function ScatterRegressionLine({ regression }) {
  const xScale = useXScale();
  const yScale = useYScale();
  const clipPathId = `scatter-regression-${React.useId()}`;

  if (!regression || !xScale || !yScale) return null;
  const xMin = Number(regression.x_min);
  const xMax = Number(regression.x_max);
  const slope = Number(regression.slope);
  const intercept = Number(regression.intercept);
  const yMin = Number.isFinite(Number(regression.y_min)) ? Number(regression.y_min) : slope * xMin + intercept;
  const yMax = Number.isFinite(Number(regression.y_max)) ? Number(regression.y_max) : slope * xMax + intercept;
  if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) return null;

  const x1 = xScale(xMin);
  const x2 = xScale(xMax);
  const y1 = yScale(yMin);
  const y2 = yScale(yMax);
  if (![x1, x2, y1, y2].every(Number.isFinite)) return null;

  return (
    <React.Fragment>
      <ChartsClipPath id={clipPathId} />
      <g clipPath={`url(#${clipPathId})`}>
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke={regression.color || '#f25467'}
          strokeWidth={2}
          strokeDasharray={regression.strokeDasharray || '6 4'}
          pointerEvents="none"
        />
      </g>
    </React.Fragment>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparkline chart
// ─────────────────────────────────────────────────────────────────────────────
function SpecSparkLineChart({ spec }) {
  const chartProps = {};
  if (spec.plotType) chartProps.plotType = spec.plotType;
  if (spec.area) chartProps.area = true;
  if (spec.curve) chartProps.curve = spec.curve;
  if (spec.color) chartProps.color = spec.color;
  
  chartProps.showHighlight = spec.showHighlight !== undefined ? spec.showHighlight : true;
  chartProps.showTooltip = spec.showTooltip !== undefined ? spec.showTooltip : true;
  
  if (spec.baseline !== undefined) chartProps.baseline = spec.baseline;
  
  const height = getResponsiveChartHeight(spec, 64);
  const sparkData = useMemo(
    () => (spec.data || []).map(value => toFiniteNumber(value)).filter(value => Number.isFinite(value)),
    [spec.data]
  );
  
  // Custom xAxis scaleType & data if passed
  const xAxisConfig = useMemo(() => {
    if (!spec.xAxis) return undefined;
    const ax = { ...spec.xAxis };
    if (ax.data && Array.isArray(ax.data)) {
      if (ax.scaleType === 'time') {
        ax.data = ax.data.map(d => new Date(d));
      }
    }
    return ax;
  }, [spec.xAxis]);
  
  // Custom yAxis config
  const yAxisConfig = useMemo(() => {
    if (!spec.yAxis) return undefined;
    return { ...spec.yAxis };
  }, [spec.yAxis]);

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);

  return (
    <Box sx={{ width: '100%', minWidth: 220, display: 'flex', justifyContent: 'center', p: 0.5, ...animationSx }}>
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

// ─────────────────────────────────────────────────────────────────────────────
// Sankey chart
// ─────────────────────────────────────────────────────────────────────────────
function SpecHeatmapChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 420);
  const xAxis = spec.xAxis || [{ data: spec.xLabels || [] }];
  const yAxis = spec.yAxis || [{ data: spec.yLabels || [] }];
  const series = (spec.series || []).map((serie, index) => ({
    id: serie.id || `heatmap-${index}`,
    label: serie.label || spec.title || 'Heatmap',
    data: serie.data || [],
    valueFormatter: (value) => value == null ? '' : toFiniteNumber(value).toFixed(3),
    ...serie,
  }));

  if (!series.length || !series.some((serie) => Array.isArray(serie.data) && serie.data.length)) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No heatmap data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <Heatmap
        height={height}
        xAxis={xAxis}
        yAxis={yAxis}
        series={series}
        borderRadius={spec.borderRadius ?? 4}
        hideLegend={spec.hideLegend ?? false}
        margin={spec.margin || { top: 24, right: 36, bottom: 58, left: 72 }}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsLegend-root': { color: '#d1d5db' },
        }}
      />
    </Box>
  );
}

function SpecPremiumSankeyChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const nodes = spec.nodes || [];
  const links = spec.links || [];
  const formatValue = getValueFormatter(spec.valueFormatter || 'none');

  if (!links.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No Sankey data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <SankeyChart
        height={height}
        series={{
          data: { nodes, links },
          nodeOptions: { showLabels: true, ...(spec.nodeOptions || {}) },
          linkOptions: { opacity: 0.42, color: 'source', ...(spec.linkOptions || {}) },
          valueFormatter: (value) => formatValue(value),
        }}
        margin={spec.margin || { top: 24, right: 24, bottom: 24, left: 24 }}
      />
    </Box>
  );
}

function SpecFunnelChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const series = spec.series || [];
  if (!series.length || !series.some((serie) => Array.isArray(serie.data) && serie.data.length)) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No funnel data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <FunnelChart
        height={height}
        series={series}
        hideLegend={spec.hideLegend ?? false}
        margin={spec.margin || { top: 24, right: 24, bottom: 30, left: 24 }}
      />
    </Box>
  );
}

function SpecRadarChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const metrics = spec.radar?.metrics || spec.metrics || [];
  const series = spec.series || [];
  if (!metrics.length || !series.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No radar data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadarChart
        height={height}
        radar={{ metrics }}
        series={series}
        hideLegend={spec.hideLegend ?? false}
        margin={spec.margin || { top: 24, right: 28, bottom: 30, left: 28 }}
      />
    </Box>
  );
}

function SpecGaugeChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 260);
  const value = toFiniteNumber(spec.value);
  const valueMin = toFiniteNumber(spec.valueMin, 0);
  const valueMax = toFiniteNumber(spec.valueMax, 100);

  return (
    <Box sx={{ width: '100%', minWidth: 220, display: 'flex', justifyContent: 'center' }}>
      <Gauge
        width={Math.min(420, spec.width || 360)}
        height={height}
        value={value}
        valueMin={valueMin}
        valueMax={valueMax}
        startAngle={spec.startAngle ?? -110}
        endAngle={spec.endAngle ?? 110}
        text={spec.text || (({ value: gaugeValue }) => `${toFiniteNumber(gaugeValue).toFixed(0)}`)}
        sx={{
          '& .MuiGauge-valueText': { fill: '#f8fafc', fontSize: 28, fontWeight: 700 },
          '& .MuiGauge-referenceArc': { fill: '#1f2937' },
          '& .MuiGauge-valueArc': { fill: PALETTE[0] },
        }}
      />
    </Box>
  );
}

function SpecRadialBarChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const categories = spec.categories || [];
  const series = spec.series || [];
  if (!categories.length || !series.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No radial bar data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadialBarChart
        height={height}
        series={series}
        rotationAxis={[{ data: categories, scaleType: 'band' }]}
        radiusAxis={[{ scaleType: 'linear' }]}
        grid={spec.grid || { radius: true, rotation: true }}
        hideLegend={spec.hideLegend ?? false}
      />
    </Box>
  );
}

function SpecRadialLineChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const categories = spec.categories || [];
  const series = spec.series || [];
  if (!categories.length || !series.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No radial line data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadialLineChart
        height={height}
        series={series}
        rotationAxis={[{ data: categories, scaleType: 'point' }]}
        radiusAxis={[{ scaleType: 'linear' }]}
        grid={spec.grid || { radius: true, rotation: true }}
        hideLegend={spec.hideLegend ?? false}
      />
    </Box>
  );
}

function SpecSankeyChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 350);
  const containerRef = React.useRef(null);
  const [containerWidth, setContainerWidth] = React.useState(360);
  const [activeId, setActiveId] = React.useState(null);
  const formatValue = useMemo(() => getValueFormatter(spec.valueFormatter || 'none'), [spec.valueFormatter]);

  React.useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) {
        setContainerWidth(Math.max(320, entries[0].contentRect.width || 640));
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  const layout = useMemo(() => {
    const nodeWidth = spec.nodeOptions?.width ?? 12;
    const nodePadding = spec.nodeOptions?.padding ?? 18;
    const margin = { top: 34, right: 28, bottom: 28, left: 28 };
    const innerWidth = Math.max(260, containerWidth - margin.left - margin.right);
    const innerHeight = Math.max(180, height - margin.top - margin.bottom);
    const nodesById = new Map();

    (spec.nodes || []).forEach((node, index) => {
      const id = node.id ?? node.label ?? `node-${index}`;
      nodesById.set(id, {
        id,
        label: node.label || String(id),
        color: node.color || PALETTE[index % PALETTE.length],
      });
    });

    const links = (spec.links || [])
      .map((link, index) => ({
        id: `${link.source}-${link.target}-${index}`,
        source: link.source,
        target: link.target,
        value: Math.max(0, toFiniteNumber(link.value)),
      }))
      .filter((link) => link.source != null && link.target != null && link.value > 0);

    links.forEach((link) => {
      if (!nodesById.has(link.source)) {
        nodesById.set(link.source, {
          id: link.source,
          label: String(link.source),
          color: PALETTE[nodesById.size % PALETTE.length],
        });
      }
      if (!nodesById.has(link.target)) {
        nodesById.set(link.target, {
          id: link.target,
          label: String(link.target),
          color: PALETTE[nodesById.size % PALETTE.length],
        });
      }
    });

    const nodes = Array.from(nodesById.values());
    const incoming = new Map(nodes.map((node) => [node.id, 0]));
    const outgoing = new Map(nodes.map((node) => [node.id, 0]));
    links.forEach((link) => {
      outgoing.set(link.source, (outgoing.get(link.source) || 0) + link.value);
      incoming.set(link.target, (incoming.get(link.target) || 0) + link.value);
    });

    const depths = new Map(nodes.map((node) => [node.id, incoming.get(node.id) ? 1 : 0]));
    for (let i = 0; i < nodes.length; i += 1) {
      links.forEach((link) => {
        depths.set(link.target, Math.max(depths.get(link.target) || 0, (depths.get(link.source) || 0) + 1));
      });
    }
    const maxDepth = Math.max(1, ...depths.values());

    const groups = new Map();
    nodes.forEach((node) => {
      const depth = depths.get(node.id) || 0;
      if (!groups.has(depth)) groups.set(depth, []);
      groups.get(depth).push(node);
    });

    const valueByNode = new Map(nodes.map((node) => [
      node.id,
      Math.max(incoming.get(node.id) || 0, outgoing.get(node.id) || 0, 1),
    ]));
    const maxGroupTotal = Math.max(
      1,
      ...Array.from(groups.values()).map((group) =>
        group.reduce((sum, node) => sum + valueByNode.get(node.id), 0)
      )
    );
    const maxGroupCount = Math.max(1, ...Array.from(groups.values()).map((group) => group.length));
    const valueScale = Math.max(
      0.0001,
      (innerHeight - nodePadding * Math.max(0, maxGroupCount - 1)) / maxGroupTotal
    );

    const positionedNodes = new Map();
    groups.forEach((group, depth) => {
      const groupHeight =
        group.reduce((sum, node) => sum + valueByNode.get(node.id) * valueScale, 0) +
        nodePadding * Math.max(0, group.length - 1);
      let y = margin.top + Math.max(0, (innerHeight - groupHeight) / 2);
      group.forEach((node) => {
        const nodeHeight = Math.max(16, valueByNode.get(node.id) * valueScale);
        const x = margin.left + (innerWidth - nodeWidth) * (depth / maxDepth);
        positionedNodes.set(node.id, {
          ...node,
          value: valueByNode.get(node.id),
          x0: x,
          x1: x + nodeWidth,
          y0: y,
          y1: y + nodeHeight,
          depth,
        });
        y += nodeHeight + nodePadding;
      });
    });

    const sourceOffsets = new Map(nodes.map((node) => [node.id, 0]));
    const targetOffsets = new Map(nodes.map((node) => [node.id, 0]));
    const positionedLinks = links.map((link) => {
      const source = positionedNodes.get(link.source);
      const target = positionedNodes.get(link.target);
      const width = Math.max(2, link.value * valueScale);
      const sourceY = source.y0 + (sourceOffsets.get(link.source) || 0) + width / 2;
      const targetY = target.y0 + (targetOffsets.get(link.target) || 0) + width / 2;
      sourceOffsets.set(link.source, (sourceOffsets.get(link.source) || 0) + width);
      targetOffsets.set(link.target, (targetOffsets.get(link.target) || 0) + width);
      const midX = (source.x1 + target.x0) / 2;
      return {
        ...link,
        source,
        target,
        width,
        color: target.color,
        d: `M ${source.x1} ${sourceY} C ${midX} ${sourceY}, ${midX} ${targetY}, ${target.x0} ${targetY}`,
      };
    });

    return {
      width: containerWidth,
      height,
      nodeWidth,
      nodes: Array.from(positionedNodes.values()),
      links: positionedLinks,
    };
  }, [containerWidth, height, spec.links, spec.nodeOptions, spec.nodes]);

  if (!layout.nodes.length || !layout.links.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No Sankey data available.</Typography>
      </Box>
    );
  }

  return (
    <Box ref={containerRef} sx={{ width: '100%', height, minHeight: height, p: 0.5, position: 'relative' }}>
      <svg width="100%" height={height} viewBox={`0 0 ${layout.width} ${layout.height}`} role="img" aria-label={spec.title || 'Sankey chart'}>
        <g fill="none">
          {layout.links.map((link) => {
            const muted = activeId && activeId !== link.id && activeId !== link.source.id && activeId !== link.target.id;
            return (
              <path
                key={link.id}
                d={link.d}
                stroke={link.color}
                strokeWidth={link.width}
                strokeOpacity={muted ? 0.18 : 0.56}
                strokeLinecap="butt"
                onMouseEnter={() => setActiveId(link.id)}
                onMouseLeave={() => setActiveId(null)}
              >
                <title>{`${link.source.label} to ${link.target.label}: ${formatValue(link.value)}`}</title>
              </path>
            );
          })}
        </g>
        <g>
          {layout.nodes.map((node) => {
            const muted = activeId && activeId !== node.id && !String(activeId).startsWith(`${node.id}-`);
            const isLeft = node.depth === 0;
            const labelX = isLeft ? node.x1 + 10 : node.x0 - 10;
            return (
              <g
                key={node.id}
                onMouseEnter={() => setActiveId(node.id)}
                onMouseLeave={() => setActiveId(null)}
                opacity={muted ? 0.55 : 1}
              >
                <rect
                  x={node.x0}
                  y={node.y0}
                  width={layout.nodeWidth}
                  height={node.y1 - node.y0}
                  rx="2"
                  fill={node.color}
                />
                <text
                  x={labelX}
                  y={(node.y0 + node.y1) / 2 - 5}
                  textAnchor={isLeft ? 'start' : 'end'}
                  dominantBaseline="middle"
                  fill="#f8fafc"
                  fontSize="12"
                  fontWeight="700"
                >
                  {node.label}
                </text>
                <text
                  x={labelX}
                  y={(node.y0 + node.y1) / 2 + 11}
                  textAnchor={isLeft ? 'start' : 'end'}
                  dominantBaseline="middle"
                  fill="#cbd5e1"
                  fontSize="11"
                >
                  {formatValue(node.value)}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Candlestick chart
// ─────────────────────────────────────────────────────────────────────────────
function SpecCandlestickChart({ spec }) {
  const height = getResponsiveChartHeight(spec, CHART_HEIGHT);
  const containerRef = React.useRef(null);
  const [containerWidth, setContainerWidth] = React.useState(360);
  const [hoveredIndex, setHoveredIndex] = React.useState(null);
  const primarySeries = Array.isArray(spec?.series)
    ? spec.series.find((series) => Array.isArray(series?.data) && series.data.length > 0)
    : null;
  const pts = primarySeries?.data || (Array.isArray(spec?.data) ? spec.data : []);

  const formatVolume = (val) => {
    if (val == null) return '';
    if (val >= 1000000000) return `${(val / 1000000000).toFixed(1)}B`;
    if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
    if (val >= 1000) return `${(val / 1000).toFixed(1)}k`;
    return val.toString();
  };

  const formatAsDollar = (value) => {
    if (value == null) return '';
    return `$${value.toLocaleString('en-US', { maximumFractionDigits: value >= 100 ? 0 : 2 })}`;
  };

  React.useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) {
        setContainerWidth(Math.max(320, entries[0].contentRect.width || 640));
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  const chart = useMemo(() => {
    const data = pts
      .map((entry, index) => ({
        index,
        date: entry.date,
        dateObj: entry.date ? new Date(entry.date) : null,
        open: toFiniteNumber(entry.open),
        high: toFiniteNumber(entry.high),
        low: toFiniteNumber(entry.low),
        close: toFiniteNumber(entry.close),
        volume: Math.max(0, toFiniteNumber(entry.volume)),
      }))
      .sort((a, b) => {
        const aTime = a.dateObj instanceof Date && !Number.isNaN(a.dateObj.getTime()) ? a.dateObj.getTime() : 0;
        const bTime = b.dateObj instanceof Date && !Number.isNaN(b.dateObj.getTime()) ? b.dateObj.getTime() : 0;
        return aTime - bTime;
      })
      .map((entry, index) => ({ ...entry, index }));

    const margin = { top: 22, right: 64, bottom: 42, left: 34 };
    const innerWidth = Math.max(260, containerWidth - margin.left - margin.right);
    const innerHeight = Math.max(200, height - margin.top - margin.bottom);
    const volumeHeight = data.some((entry) => entry.volume > 0) ? Math.max(46, innerHeight * 0.22) : 0;
    const volumeGap = volumeHeight ? 12 : 0;
    const priceHeight = innerHeight - volumeHeight - volumeGap;
    const plotBottom = margin.top + priceHeight;

    const minLow = Math.min(...data.map((entry) => entry.low));
    const maxHigh = Math.max(...data.map((entry) => entry.high));
    const pricePadding = Math.max((maxHigh - minLow) * 0.1, 1);
    const minPrice = minLow - pricePadding;
    const maxPrice = maxHigh + pricePadding;
    const priceRange = Math.max(maxPrice - minPrice, 1);
    const maxVolume = Math.max(1, ...data.map((entry) => entry.volume));
    const step = innerWidth / Math.max(1, data.length);
    const candleWidth = Math.max(5, Math.min(16, step * 0.58));
    const denseMode = data.length > 60;

    const xFor = (index) => margin.left + step * (index + 0.5);
    const priceY = (value) => margin.top + ((maxPrice - value) / priceRange) * priceHeight;
    const volumeY = (value) => plotBottom + volumeGap + volumeHeight - (value / maxVolume) * volumeHeight;

    const priceTicks = Array.from({ length: 5 }, (_, index) => {
      const value = minPrice + (priceRange * index) / 4;
      return { value, y: priceY(value) };
    }).reverse();

    const windowSize = 20;
    const movingAverage = data.map((_, index) => {
      if (index < windowSize - 1) return null;
      const window = data.slice(index - windowSize + 1, index + 1);
      return window.reduce((sum, entry) => sum + entry.close, 0) / window.length;
    });
    const movingAveragePath = movingAverage
      .map((value, index) => value == null ? null : `${index === windowSize - 1 ? 'M' : 'L'} ${xFor(index)} ${priceY(value)}`)
      .filter(Boolean)
      .join(' ');

    const tickEvery = Math.max(1, Math.ceil(data.length / 7));
    const dateTicks = data
      .filter((_, index) => index === 0 || index === data.length - 1 || index % tickEvery === 0)
      .map((entry) => ({
        x: xFor(entry.index),
        label: entry.dateObj instanceof Date && !Number.isNaN(entry.dateObj.getTime())
          ? entry.dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
          : entry.date,
      }));

    return {
      width: containerWidth,
      height,
      data,
      margin,
      innerWidth,
      priceHeight,
      volumeHeight,
      volumeGap,
      plotBottom,
      candleWidth,
      xFor,
      priceY,
      volumeY,
      priceTicks,
      dateTicks,
      movingAverage,
      movingAveragePath,
      hasVolume: volumeHeight > 0,
      denseMode,
    };
  }, [containerWidth, height, pts]);

  if (pts.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No candlestick data available.</Typography>
      </Box>
    );
  }

  const hovered = hoveredIndex == null ? null : chart.data[hoveredIndex];
  const tooltipLeft = hovered ? Math.min(Math.max(chart.xFor(hovered.index) + 12, 12), chart.width - 190) : 0;
  const handleHoverMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * chart.width;
    const step = chart.width / Math.max(1, chart.data.length);
    const index = Math.floor((relativeX - chart.margin.left) / Math.max(step, 1));
    if (index >= 0 && index < chart.data.length) setHoveredIndex(index);
    else setHoveredIndex(null);
  };

  return (
    <Box ref={containerRef} sx={{ width: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 0.5, mb: 0.5 }}>
        <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 600, letterSpacing: 0.2 }}>
          OHLC
        </Typography>
        {chart.denseMode && (
          <Chip
            label="Dense data"
            size="small"
            variant="outlined"
            sx={{
              height: 22,
              borderColor: '#334155',
              color: '#cbd5e1',
              bgcolor: 'rgba(15, 23, 42, 0.55)',
              '& .MuiChip-label': { px: 1, fontSize: 11, fontWeight: 600 },
            }}
          />
        )}
      </Box>
      <svg width="100%" height={height} viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label={spec.title || 'Candlestick chart'}>
        <rect x="0" y="0" width={chart.width} height={chart.height} fill="transparent" />
        <g>
          {chart.priceTicks.map((tick) => (
            <g key={tick.value}>
              <line
                x1={chart.margin.left}
                x2={chart.margin.left + chart.innerWidth}
                y1={tick.y}
                y2={tick.y}
                stroke="#374151"
                strokeDasharray="4 4"
              />
              <text x={chart.margin.left + chart.innerWidth + 8} y={tick.y + 4} fill="#cbd5e1" fontSize="11" fontWeight="700">
                {formatAsDollar(tick.value)}
              </text>
            </g>
          ))}
          {chart.dateTicks.map((tick) => (
            <g key={`${tick.x}-${tick.label}`}>
              <line
                x1={tick.x}
                x2={tick.x}
                y1={chart.margin.top}
                y2={chart.hasVolume ? chart.plotBottom + chart.volumeGap + chart.volumeHeight : chart.plotBottom}
                stroke="#334155"
                strokeOpacity="0.55"
              />
              <text x={tick.x} y={height - 14} fill="#cbd5e1" fontSize="11" textAnchor="middle">
                {tick.label}
              </text>
            </g>
          ))}
        </g>
        {chart.hasVolume && (
          <g>
            <line
              x1={chart.margin.left}
              x2={chart.margin.left + chart.innerWidth}
              y1={chart.plotBottom + chart.volumeGap + chart.volumeHeight}
              y2={chart.plotBottom + chart.volumeGap + chart.volumeHeight}
              stroke="#475569"
            />
            {chart.data.map((entry) => {
              const rising = entry.index === 0 || entry.close >= chart.data[entry.index - 1].close;
              const x = chart.xFor(entry.index) - chart.candleWidth / 2;
              const y = chart.volumeY(entry.volume);
              const h = chart.plotBottom + chart.volumeGap + chart.volumeHeight - y;
              return (
                <rect
                  key={`volume-${entry.index}`}
                  x={x}
                  y={y}
                  width={chart.candleWidth}
                  height={Math.max(1, h)}
                  fill={rising ? '#22c55e' : '#ef4444'}
                  opacity="0.78"
                />
              );
            })}
          </g>
        )}
        <g>
          {chart.movingAveragePath && (
            <path d={chart.movingAveragePath} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" />
          )}
          {chart.data.map((entry) => {
            const rising = entry.close >= entry.open;
            const color = rising ? '#22c55e' : '#ef4444';
            const x = chart.xFor(entry.index);
            const openY = chart.priceY(entry.open);
            const closeY = chart.priceY(entry.close);
            const highY = chart.priceY(entry.high);
            const lowY = chart.priceY(entry.low);
            const bodyY = Math.min(openY, closeY);
            const bodyHeight = Math.max(2, Math.abs(closeY - openY));
            return (
              <g
                key={`candle-${entry.index}`}
                pointerEvents="none"
              >
                <line x1={x} x2={x} y1={highY} y2={lowY} stroke={color} strokeWidth="1.5" />
                <rect
                  x={x - chart.candleWidth / 2}
                  y={bodyY}
                  width={chart.candleWidth}
                  height={bodyHeight}
                  rx="1"
                  fill={color}
                />
                <rect
                  x={x - chart.candleWidth / 2 - 3}
                  y={chart.margin.top}
                  width={chart.candleWidth + 6}
                  height={chart.priceHeight + chart.volumeGap + chart.volumeHeight}
                  fill="transparent"
                >
                  <title>{`${entry.date}: O ${formatAsDollar(entry.open)} H ${formatAsDollar(entry.high)} L ${formatAsDollar(entry.low)} C ${formatAsDollar(entry.close)} V ${formatVolume(entry.volume)}`}</title>
                </rect>
              </g>
            );
          })}
        </g>
        <rect
          x={chart.margin.left}
          y={chart.margin.top}
          width={chart.innerWidth}
          height={chart.priceHeight + chart.volumeGap + chart.volumeHeight}
          fill="transparent"
          onMouseMove={handleHoverMove}
          onMouseLeave={() => setHoveredIndex(null)}
          style={{ cursor: 'crosshair' }}
        />
        <g>
          <line x1={chart.margin.left} x2={chart.margin.left + chart.innerWidth} y1={chart.plotBottom} y2={chart.plotBottom} stroke="#94a3b8" />
          <line x1={chart.margin.left + chart.innerWidth} x2={chart.margin.left + chart.innerWidth} y1={chart.margin.top} y2={chart.plotBottom} stroke="#94a3b8" />
          {chart.movingAveragePath && (
            <g>
              <line x1={chart.margin.left + chart.innerWidth - 110} x2={chart.margin.left + chart.innerWidth - 82} y1={chart.margin.top + 10} y2={chart.margin.top + 10} stroke="#3b82f6" strokeWidth="2" />
              <text x={chart.margin.left + chart.innerWidth - 76} y={chart.margin.top + 14} fill="#cbd5e1" fontSize="11">20-day SMA</text>
            </g>
          )}
        </g>
      </svg>
      {hovered && (
        <Box
          sx={{
            position: 'absolute',
            top: 12,
            left: tooltipLeft,
            pointerEvents: 'none',
            background: 'rgba(15, 23, 42, 0.92)',
            border: '1px solid #334155',
            borderRadius: 1,
            color: '#e5e7eb',
            px: 1,
            py: 0.75,
            fontSize: 11,
            boxShadow: '0 10px 24px rgba(0,0,0,0.25)',
            zIndex: 2,
          }}
        >
          <Box sx={{ fontWeight: 700, mb: 0.25 }}>{hovered.date}</Box>
          <Box>O {formatAsDollar(hovered.open)} H {formatAsDollar(hovered.high)}</Box>
          <Box>L {formatAsDollar(hovered.low)} C {formatAsDollar(hovered.close)}</Box>
          {chart.hasVolume && <Box>V {formatVolume(hovered.volume)}</Box>}
          {chart.movingAverage[hovered.index] != null && <Box sx={{ color: '#93c5fd' }}>SMA {formatAsDollar(chart.movingAverage[hovered.index])}</Box>}
        </Box>
      )}
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Network Chart Component
// ─────────────────────────────────────────────────────────────────────────────
function SpecNetworkChart({ spec }) {
  const height = spec.height !== undefined ? spec.height : 400;
  const nodes = spec.nodes || [];
  const edges = spec.edges || [];

  const containerRef = React.useRef(null);
  const [containerWidth, setContainerWidth] = React.useState(600);
  const [hoveredNodeId, setHoveredNodeId] = React.useState(null);
  const [tooltipPos, setTooltipPos] = React.useState(null);

  React.useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries && entries[0]) {
        setContainerWidth(entries[0].contentRect.width || 600);
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  const margin = { top: 40, right: 60, bottom: 40, left: 60 };
  const plotWidth = containerWidth - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const { minX, maxX, minY, maxY } = useMemo(() => {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    Object.values(spec.node_positions || {}).forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });
    return {
      minX: minX === Infinity ? -1 : minX,
      maxX: maxX === -Infinity ? 1 : maxX,
      minY: minY === Infinity ? -1 : minY,
      maxY: maxY === -Infinity ? 1 : maxY,
    };
  }, [spec.node_positions]);

  const nodeCoords = useMemo(() => {
    const coords = {};
    const rx = maxX - minX || 1;
    const ry = maxY - minY || 1;
    
    nodes.forEach(node => {
      const pos = spec.node_positions?.[node.id] || [0, 0];
      const px = margin.left + ((pos[0] - minX) / rx) * plotWidth;
      const py = margin.top + ((pos[1] - minY) / ry) * plotHeight;
      coords[node.id] = { x: px, y: py };
    });
    return coords;
  }, [nodes, spec.node_positions, minX, maxX, minY, maxY, plotWidth, plotHeight, margin]);

  // Highlight connections map
  const connectedNodes = useMemo(() => {
    if (!hoveredNodeId) return new Set();
    const set = new Set([hoveredNodeId]);
    edges.forEach(edge => {
      if (edge.source === hoveredNodeId) set.add(edge.target);
      if (edge.target === hoveredNodeId) set.add(edge.source);
    });
    return set;
  }, [hoveredNodeId, edges]);

  const handleNodeMouseEnter = (nodeId, e) => {
    setHoveredNodeId(nodeId);
    
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const tooltipWidth = 200;
    let tooltipX = mouseX + 15;
    let tooltipY = mouseY - 40;

    if (tooltipX + tooltipWidth > containerWidth) {
      tooltipX = mouseX - tooltipWidth - 15;
    }
    if (tooltipY < 0) {
      tooltipY = 10;
    }

    setTooltipPos({ x: tooltipX, y: tooltipY });
  };

  const handleNodeMouseLeave = () => {
    setHoveredNodeId(null);
  };

  const activeNodeDetails = useMemo(() => {
    if (!hoveredNodeId) return null;
    const node = nodes.find(n => n.id === hoveredNodeId);
    if (!node) return null;

    const connections = [];
    edges.forEach(edge => {
      if (edge.source === hoveredNodeId) {
        connections.push({ name: edge.target, weight: edge.weight });
      } else if (edge.target === hoveredNodeId) {
        connections.push({ name: edge.source, weight: edge.weight });
      }
    });

    connections.sort((a, b) => b.weight - a.weight);

    return {
      node,
      connections,
    };
  }, [hoveredNodeId, nodes, edges]);

  if (nodes.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No network data available.</Typography>
      </Box>
    );
  }

  const animationSx = getAnimationSx(spec.animation);

  return (
    <Box
      ref={containerRef}
      sx={{ width: '100%', height, minHeight: height, position: 'relative', overflow: 'hidden', ...animationSx }}
    >
      <svg width={containerWidth} height={height} style={{ display: 'block' }}>
        {/* Render Edges */}
        {edges.map((edge, idx) => {
          const from = nodeCoords[edge.source];
          const to = nodeCoords[edge.target];
          if (!from || !to) return null;

          const isHighlighted = !hoveredNodeId || (edge.source === hoveredNodeId || edge.target === hoveredNodeId);
          const strokeColor = isHighlighted ? '#a5f3fc' : '#374151'; // cyan light vs dark gray
          const strokeWidth = isHighlighted ? 2.0 : 1.0;
          const opacity = isHighlighted ? 0.7 : 0.15;

          return (
            <line
              key={`edge-${idx}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              opacity={opacity}
              style={{ transition: 'all 0.2s ease' }}
            />
          );
        })}

        {/* Render Nodes */}
        {nodes.map(node => {
          const coords = nodeCoords[node.id];
          if (!coords) return null;

          const isStock = node.is_stock;
          const radius = isStock ? (16 + 18 * (node.risk_score || 0.0)) : 10;
          
          const isHighlighted = !hoveredNodeId || connectedNodes.has(node.id);
          const opacity = isHighlighted ? 1.0 : 0.25;

          const nodeColor = isStock ? '#22d3ee' : '#f59e0b'; // Cyan for Stock, Orange/Amber for Holder
          const nodeBorder = '#e5e7eb';

          return (
            <g
              key={`node-${node.id}`}
              onMouseEnter={(e) => handleNodeMouseEnter(node.id, e)}
              onMouseLeave={handleNodeMouseLeave}
              style={{ cursor: 'pointer', transition: 'all 0.2s ease', opacity }}
            >
              {/* Radial gradient or shadow effect for glowing stocks */}
              {isStock && (
                <circle
                  cx={coords.x}
                  cy={coords.y}
                  r={radius + 4}
                  fill={nodeColor}
                  opacity={hoveredNodeId === node.id ? 0.4 : 0.15}
                  style={{ transition: 'all 0.2s ease' }}
                />
              )}
              <circle
                cx={coords.x}
                cy={coords.y}
                r={radius}
                fill={nodeColor}
                stroke={nodeBorder}
                strokeWidth={isStock ? 1.5 : 1.0}
              />
              {isStock ? (
                // Draw Ticker Label inside circle
                <text
                  x={coords.x}
                  y={coords.y + 4}
                  textAnchor="middle"
                  fill="#111827"
                  fontSize={10}
                  fontWeight="bold"
                  pointerEvents="none"
                >
                  {node.id}
                </text>
              ) : (
                // Draw Holder Label above circle
                <text
                  x={coords.x}
                  y={coords.y - 14}
                  textAnchor="middle"
                  fill="#e5e7eb"
                  fontSize={9}
                  fontWeight={hoveredNodeId === node.id ? 'bold' : 'normal'}
                  pointerEvents="none"
                  style={{ transition: 'all 0.2s ease' }}
                >
                  {node.id.length > 20 ? `${node.id.substring(0, 18)}...` : node.id}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Node Tooltip */}
      {hoveredNodeId !== null && tooltipPos && activeNodeDetails && (
        <Box
          sx={{
            position: 'absolute',
            left: tooltipPos.x,
            top: tooltipPos.y,
            pointerEvents: 'none',
            zIndex: 10,
            bgcolor: 'rgba(17, 24, 39, 0.95)',
            border: '1px solid #374151',
            borderRadius: '6px',
            p: 1.25,
            boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -2px rgba(0,0,0,0.05)',
            backdropFilter: 'blur(4px)',
            minWidth: 200,
            maxWidth: 260,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              color: '#e5e7eb',
              display: 'block',
              fontWeight: 600,
              mb: 0.5,
              borderBottom: '1px solid #374151',
              pb: 0.5,
            }}
          >
            {activeNodeDetails.node.is_stock ? `Stock Ticker: ${activeNodeDetails.node.id}` : `Institution: ${activeNodeDetails.node.id}`}
          </Typography>
          <Box sx={{ fontSize: 11, color: '#9ca3af' }}>
            {activeNodeDetails.node.is_stock && (
              <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between' }}>
                <span>Systemic Risk Score:</span>
                <span style={{ color: '#22d3ee', fontWeight: 600 }}>
                  {(activeNodeDetails.node.risk_score * 100).toFixed(2)}%
                </span>
              </Box>
            )}
            
            <Typography variant="caption" sx={{ color: '#e5e7eb', display: 'block', fontWeight: 600, mt: 0.5, mb: 0.25 }}>
              {activeNodeDetails.node.is_stock ? 'Top Institutional Holders:' : 'Investments:'}
            </Typography>
            {activeNodeDetails.connections.length > 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {activeNodeDetails.connections.slice(0, 5).map((conn, idx) => (
                  <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
                    <span style={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      maxWidth: 130
                    }}>
                      {conn.name}
                    </span>
                    <span style={{ color: '#f59e0b', fontWeight: 500 }}>
                      {(conn.weight * 100).toFixed(2)}%
                    </span>
                  </Box>
                ))}
                {activeNodeDetails.connections.length > 5 && (
                  <span style={{ color: '#6b7280', fontSize: 10 }}>
                    + {activeNodeDetails.connections.length - 5} more connections
                  </span>
                )}
              </Box>
            ) : (
              <span style={{ color: '#6b7280' }}>No connection data</span>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────
export default function InlineChart({ plotId }) {
  const [spec, setSpec] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    if (!plotId) return;
    
    let isMounted = true;
    setLoading(true);
    setError('');
    
    fetch(`${BACKEND_BASE}/api/plots/${plotId}`)
      .then(async res => {
        if (!res.ok) {
          let detail = '';
          try {
            const payload = await res.json();
            detail = payload?.detail || '';
          } catch {
            detail = await res.text().catch(() => '');
          }
          const message = detail || (res.status === 404 ? 'Plot not found or expired' : 'Plot fetch failed');
          throw new Error(message);
        }
        return res.json();
      })
      .then(data => {
        if (isMounted) {
          setSpec(data);
          setLoading(false);
        }
      })
      .catch(err => {
        console.error("Failed to load plot:", err);
        if (isMounted) {
          setError(err.message || 'Visualization unavailable');
          setLoading(false);
        }
      });
      
    return () => { isMounted = false; };
  }, [plotId]);

  if (loading) {
    return (
      <Paper elevation={3} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #1f2937' }}>
        <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
          Loading visualization...
        </Typography>
        <Skeleton variant="rounded" height={24} sx={{ bgcolor: 'rgba(148, 163, 184, 0.12)', mb: 1.5 }} />
        <Skeleton variant="rounded" height={320} sx={{ bgcolor: 'rgba(148, 163, 184, 0.08)' }} />
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper elevation={0} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #374151' }}>
        <Typography variant="body2" sx={{ color: '#fca5a5' }}>
          Visualization unavailable: {error}
        </Typography>
      </Paper>
    );
  }

  if (!spec || !spec.plot_type) return null;

  const displayTitle = (spec.title || '').replace(/\s*\(Interactive Pro\)\s*$/i, ' (Interactive)');

  let ChartComponent;
  switch (spec.plot_type) {
    case 'line':        ChartComponent = LineChartRenderer; break;
    case 'bar':         ChartComponent = SmartBarChartRenderer;  break;
    case 'pie':         ChartComponent = PieChartRenderer;  break;
    case 'scatter':     ChartComponent = ScatterChartRenderer; break;
    case 'sparkline':   ChartComponent = SparklineChartRenderer; break;
    case 'sankey':      ChartComponent = SankeyChartRenderer; break;
    case 'candlestick': ChartComponent = CandlestickChart; break;
    case 'heatmap':     ChartComponent = HeatmapChartRenderer; break;
    case 'network':     ChartComponent = NetworkChartRenderer; break;
    case 'funnel':      ChartComponent = FunnelChartRenderer; break;
    case 'radar':       ChartComponent = RadarChartRenderer; break;
    case 'gauge':       ChartComponent = GaugeChartRenderer; break;
    case 'radial_bar':  ChartComponent = RadialBarChartRenderer; break;
    case 'radial_line': ChartComponent = RadialLineChartRenderer; break;
    default:            return null;
  }

  return (
    <Paper
      elevation={3}
      sx={{
        p: 1.5,
        mt: 1,
        mb: 1,
        width: '100%',
        bgcolor: '#111827',
        borderRadius: 2,
        border: '1px solid #1f2937',
        overflowX: 'auto',
        overflowY: 'hidden',
        maxWidth: '100%',
      }}
    >
      <Typography
        variant="subtitle2"
        align="center"
        sx={{ color: '#e5e7eb', mb: 0.5, fontWeight: 600, letterSpacing: 0.3 }}
      >
        {displayTitle}
      </Typography>
      <ChartComponent spec={spec} />
    </Paper>
  );
}


