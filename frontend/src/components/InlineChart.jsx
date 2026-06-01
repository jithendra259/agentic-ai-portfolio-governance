import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import { PieChart } from '@mui/x-charts/PieChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { SankeyChart } from '@mui/x-charts-pro/SankeyChart';
import { Box, Typography, Paper } from '@mui/material';
import { useDrawingArea, useXScale, useAnimateBarLabel } from '@mui/x-charts/hooks';
import { useTheme, alpha } from '@mui/material/styles';

// Premium Candlestick Chart imports
import { ChartsClipPath } from '@mui/x-charts-premium/ChartsClipPath';
import { Unstable_CandlestickPlot as CandlestickPlot } from '@mui/x-charts-premium/CandlestickChart';
import { LinePlot } from '@mui/x-charts-premium/LineChart';
import { BarPlot } from '@mui/x-charts-premium/BarChart';
import { ChartsXAxis } from '@mui/x-charts-premium/ChartsXAxis';
import { ChartsYAxis } from '@mui/x-charts-premium/ChartsYAxis';
import { useAxesTooltip } from '@mui/x-charts-premium/ChartsTooltip';
import { ChartsDataProviderPremium } from '@mui/x-charts-premium/ChartsDataProviderPremium';
import { ChartsWrapper } from '@mui/x-charts-premium/ChartsWrapper';
import { ChartsAxisHighlight } from '@mui/x-charts-premium/ChartsAxisHighlight';
import { ChartsGrid } from '@mui/x-charts-premium/ChartsGrid';
import { ChartsWebGLLayer } from '@mui/x-charts-premium/ChartsWebGLLayer';
import { ChartsLayerContainer } from '@mui/x-charts-premium/ChartsLayerContainer';
import { ChartsSvgLayer } from '@mui/x-charts-premium/ChartsSvgLayer';

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

function getValueFormatter(format) {
  if (format === 'percent') {
    return v => v == null ? '' : `${v.toFixed(1)}%`;
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
      entry.connectNulls = s.connectNulls ?? true;

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

  return (
    <Box sx={{ width: '100%', height: CHART_HEIGHT, ...animationSx }}>
      <LineChart
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
            labelStyle: { fill: '#e5e7eb', fontSize: 12 },
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
  const { dataset, series, xAxisConfig, yAxisConfig } = useMemo(() => {
    if (!spec?.series?.length) return { dataset: [], series: [], xAxisConfig: [], yAxisConfig: [] };

    // Collect all unique categories across all series
    const catSet = new Set();
    spec.series.forEach(s => s.data?.forEach(pt => catSet.add(pt.x)));
    const categories = Array.from(catSet);

    // Build pivoted dataset [{label: "category", Score: 10, ...}]
    const byCategory = {};
    categories.forEach(c => { byCategory[c] = { label: c }; });
    spec.series.forEach(s => {
      s.data?.forEach(pt => {
        if (byCategory[pt.x]) byCategory[pt.x][s.name] = pt.y;
      });
    });

    const dataset = categories.map(c => byCategory[c]);

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
    if (spec.xAxis && Array.isArray(spec.xAxis)) {
      xAxisConfig = spec.xAxis.map(ax => {
        const xAx = {
          ...ax,
          scaleType: ax.scaleType || 'band',
          tickLabelStyle: { ...AXIS_STYLE, angle: -30, textAnchor: 'end' },
        };
        if (spec.categoryGapRatio != null) xAx.categoryGapRatio = spec.categoryGapRatio;
        if (spec.barGapRatio != null) xAx.barGapRatio = spec.barGapRatio;
        return xAx;
      });
    } else {
      const xAx = {
        dataKey: 'label',
        scaleType: 'band',
        tickLabelStyle: { ...AXIS_STYLE, angle: -30, textAnchor: 'end' },
        label: spec.x_label || '',
      };
      if (spec.categoryGapRatio != null) xAx.categoryGapRatio = spec.categoryGapRatio;
      if (spec.barGapRatio != null) xAx.barGapRatio = spec.barGapRatio;
      xAxisConfig = [xAx];
    }

    // ── Y-axis configuration ──
    let yAxisConfig;
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      yAxisConfig = spec.yAxis.map(ax => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(ax.value_format || spec.y_format),
        domainLimit: ax.domainLimit || 'nice',
      }));
    } else {
      yAxisConfig = [{
        tickLabelStyle: AXIS_STYLE,
        label: spec.y_label || '',
        valueFormatter: getValueFormatter(spec.y_format),
        domainLimit: 'nice',
      }];
    }

    return { dataset, series, xAxisConfig, yAxisConfig };
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

  return (
    <BarChart
      dataset={dataset}
      xAxis={xAxisConfig}
      yAxis={yAxisConfig}
      series={series}
      height={CHART_HEIGHT}
      margin={{ top: 16, right: 24, left: 60, bottom: 56 }}
      grid={gridConfig}
      borderRadius={spec.borderRadius ?? 4}
      {...(slotsConfig ? { slots: slotsConfig } : {})}
      {...(spec.layout ? { layout: spec.layout } : {})}
      {...(spec.skipAnimation ? { skipAnimation: true } : {})}
      sx={{
        '& .MuiChartsGrid-line': GRID_STYLE,
        ...animationSx,
      }}
      slotProps={{ legend: { labelStyle: { fill: '#e5e7eb', fontSize: 12 } } }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Pie Center Label
// ─────────────────────────────────────────────────────────────────────────────
function PieCenterLabel({ children }) {
  const { width, height, left, top } = useDrawingArea();
  return (
    <text
      x={left + width / 2}
      y={top + height / 2}
      textAnchor="middle"
      dominantBaseline="central"
      fill="#ffffff"
      style={{ fontSize: '1.25rem', fontWeight: 'bold' }}
    >
      {children}
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
          // Fallback default format
          const multiplier = isFractional ? 100 : 1;
          return `${(v * multiplier).toFixed(1)}%`;
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
        const multiplier = isFractional ? 100 : 1;
        return `${(val * multiplier).toFixed(1)}%`;
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

  if (!mappedSeries.length) return null;

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);

  const chartProps = {};
  if (spec.skipAnimation) chartProps.skipAnimation = true;
  if (spec.hideLegend) chartProps.hideLegend = true;
  if (spec.colors && Array.isArray(spec.colors)) chartProps.colors = spec.colors;

  return (
    <PieChart
      series={mappedSeries}
      height={CHART_HEIGHT}
      margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
      sx={{
        '& .MuiPieArcLabel-root': {
          fontSize: '11px',
          fill: '#ffffff',
          fontWeight: 'bold',
        },
        ...animationSx,
      }}
      slotProps={{
        legend: {
          labelStyle: { fill: '#e5e7eb', fontSize: 11 },
          itemMarkWidth: 10,
          itemMarkHeight: 10,
          markGap: 6,
          itemGap: 10,
        },
      }}
      {...chartProps}
    >
      {spec.centerLabel && <PieCenterLabel>{spec.centerLabel}</PieCenterLabel>}
    </PieChart>
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
        x: pt.x,
        y: pt.y,
        id: pt.id !== undefined ? pt.id : `pt-${j}`,
        ...(pt.z !== undefined ? { z: pt.z } : {}),
      }));
      const entry = {
        type: 'scatter',
        label: s.label || s.name || `Series ${i + 1}`,
        data: dataPoints,
        color: s.color || PALETTE[i % PALETTE.length],
      };
      if (s.markerSize !== undefined) entry.markerSize = s.markerSize;
      
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
        valueFormatter: getValueFormatter(ax.value_format || spec.x_format),
      }));
    }
    return [{
      tickLabelStyle: AXIS_STYLE,
      label: spec.x_label || '',
      valueFormatter: getValueFormatter(spec.x_format),
    }];
  }, [spec]);

  const yAxisConfig = useMemo(() => {
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      return spec.yAxis.map(ax => ({
        ...ax,
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(ax.value_format || spec.y_format),
      }));
    }
    return [{
      tickLabelStyle: AXIS_STYLE,
      label: spec.y_label || '',
      valueFormatter: getValueFormatter(spec.y_format),
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

  if (!series.length) return null;

  const chartProps = {};
  if (spec.skipAnimation) chartProps.skipAnimation = true;
  if (spec.hideLegend) chartProps.hideLegend = true;
  if (spec.colors && Array.isArray(spec.colors)) chartProps.colors = spec.colors;
  if (spec.hitAreaRadius !== undefined) chartProps.hitAreaRadius = spec.hitAreaRadius;

  const gridConfig = spec.grid || { horizontal: true, vertical: true };
  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);

  return (
    <ScatterChart
      series={series}
      xAxis={xAxisConfig}
      yAxis={yAxisConfig}
      {...(zAxisConfig ? { zAxis: zAxisConfig } : {})}
      height={CHART_HEIGHT}
      margin={{ top: 16, right: 24, left: 60, bottom: 56 }}
      grid={gridConfig}
      sx={{
        '& .MuiChartsGrid-line': GRID_STYLE,
        ...animationSx,
      }}
      slotProps={{ legend: { labelStyle: { fill: '#e5e7eb', fontSize: 12 } } }}
      {...chartProps}
    />
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
  
  const height = spec.height !== undefined ? spec.height : 60;
  
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
    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', p: 0.5, ...animationSx }}>
      <SparkLineChart
        data={spec.data || []}
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
function SpecSankeyChart({ spec }) {
  const height = spec.height !== undefined ? spec.height : 350;

  const valueFormatter = React.useCallback(
    (value, context) => {
      const formatter = getValueFormatter(spec.valueFormatter || 'none');
      const formatted = formatter(value);
      if (context?.type === 'link') {
        return formatted;
      }
      return `${formatted} total`;
    },
    [spec.valueFormatter]
  );

  const series = useMemo(() => {
    return {
      data: {
        nodes: spec.nodes || [],
        links: spec.links || [],
      },
      nodeOptions: {
        highlight: 'links',
        fade: 'global',
        sort: 'fixed',
        width: 12,
        padding: 18,
        ...spec.nodeOptions,
      },
      linkOptions: {
        highlight: 'nodes',
        fade: 'global',
        color: 'target-gradient',
        opacity: 0.5,
        ...spec.linkOptions,
      },
      valueFormatter,
    };
  }, [spec.nodes, spec.links, spec.nodeOptions, spec.linkOptions, valueFormatter]);

  const animationSx = useMemo(() => getAnimationSx(spec.animation), [spec.animation]);

  return (
    <Box sx={{ width: '100%', height, minHeight: height, p: 0.5, ...animationSx }}>
      <SankeyChart
        series={series}
        height={height}
        sx={animationSx}
      />
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Candlestick chart
// ─────────────────────────────────────────────────────────────────────────────
function CandlestickTooltip({ hasVolume, formatVolume }) {
  const drawingArea = useDrawingArea();
  const axesTooltipData = useAxesTooltip({
    directions: ['x'],
  });

  const tooltipData = axesTooltipData?.[0];

  if (!tooltipData) {
    return null;
  }

  const ohlcItem = tooltipData.seriesItems.find(
    (item) => item.seriesId === 'ohlc'
  );
  const movingAverageItem = tooltipData.seriesItems.find(
    (item) => item.seriesId === 'moving-average'
  );
  const volumeItem = tooltipData.seriesItems.find(
    (item) => item.seriesId === 'volume'
  );

  const formatVal = (v) => v == null ? '' : v.toFixed(2);

  const ohlcValue = ohlcItem?.value;
  const maValue = movingAverageItem?.value;
  const volValue = volumeItem?.value;

  return (
    <foreignObject
      x={drawingArea.left}
      y={drawingArea.top}
      width={drawingArea.width}
      height={drawingArea.height}
      style={{ pointerEvents: 'none' }}
    >
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        fontSize: '11px',
        padding: '6px 8px',
        color: '#e5e7eb',
        background: 'rgba(17, 24, 39, 0.85)',
        backdropFilter: 'blur(4px)',
        border: '1px solid #374151',
        borderRadius: '4px',
        width: 'fit-content',
        margin: '8px',
        boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
      }}>
        {ohlcValue && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <span><strong style={{ color: '#9ca3af' }}>O:</strong> ${formatVal(ohlcValue.open)}</span>
            <span><strong style={{ color: '#9ca3af' }}>H:</strong> ${formatVal(ohlcValue.high)}</span>
            <span><strong style={{ color: '#9ca3af' }}>L:</strong> ${formatVal(ohlcValue.low)}</span>
            <span><strong style={{ color: '#9ca3af' }}>C:</strong> ${formatVal(ohlcValue.close)}</span>
            {hasVolume && volValue != null && (
              <span><strong style={{ color: '#9ca3af' }}>V:</strong> {formatVolume(volValue)}</span>
            )}
          </div>
        )}
        {maValue != null && (
          <div>
            <span style={{ color: '#3b82f6', fontWeight: 600 }}>20-day MA:</span> ${formatVal(maValue)}
          </div>
        )}
      </div>
    </foreignObject>
  );
}

function SpecCandlestickChart({ spec }) {
  const height = spec.height !== undefined ? spec.height : CHART_HEIGHT;
  const series = spec.series;
  const pts = series?.[0]?.data || [];

  const theme = useTheme();
  const clipId = React.useId();
  const clipPathId = `clip-path-${clipId.replace(/:/g, '')}`;

  const formatVolume = (val) => {
    if (val == null) return '';
    if (val >= 1000000000) return `${(val / 1000000000).toFixed(1)}B`;
    if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
    if (val >= 1000) return `${(val / 1000).toFixed(1)}k`;
    return val.toString();
  };

  const formatAsDollar = (value) => {
    if (value == null) return '';
    return `$${value.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
  };

  const { maxVolume, hasVolume } = useMemo(() => {
    let maxVol = 0;
    let hasVol = false;
    pts.forEach(pt => {
      if (pt.volume != null && pt.volume > 0) {
        hasVol = true;
        if (pt.volume > maxVol) maxVol = pt.volume;
      }
    });
    return { maxVolume: maxVol, hasVolume: hasVol };
  }, [pts]);

  const xData = useMemo(() => pts.map((entry) => new Date(entry.date)), [pts]);

  const ohlcData = useMemo(() => pts.map((entry) => [
    entry.open,
    entry.high,
    entry.low,
    entry.close,
  ]), [pts]);

  const volumeData = useMemo(() => pts.map((entry) => entry.volume || 0), [pts]);

  const movingAverageData = useMemo(() => {
    const windowSize = 20;
    return pts.map((_, i) => {
      if (i < windowSize - 1) {
        return null;
      }
      const sum = pts
        .slice(i - windowSize + 1, i + 1)
        .reduce((acc, entry) => acc + entry.close, 0);
      return sum / windowSize;
    });
  }, [pts]);

  const volumeBarColorGetter = ({ dataIndex }) => {
    if (dataIndex === 0) {
      return theme.palette.success.main;
    }
    return pts[dataIndex].close >= pts[dataIndex - 1].close
      ? theme.palette.success.main
      : theme.palette.error.main;
  };

  if (pts.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No candlestick data available.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <ChartsDataProviderPremium
        series={[
          {
            id: 'ohlc',
            type: 'ohlc',
            data: ohlcData,
            label: 'Price',
          },
          {
            id: 'moving-average',
            type: 'line',
            data: movingAverageData,
            label: '20-day SMA',
            color: '#3b82f6',
          },
          ...(hasVolume ? [
            {
              id: 'volume',
              type: 'bar',
              data: volumeData,
              label: 'Volume',
              colorGetter: volumeBarColorGetter,
              yAxisId: 'volume',
            }
          ] : []),
        ]}
        xAxis={[
          {
            data: xData,
            scaleType: 'band',
            valueFormatter: (value) =>
              value instanceof Date
                ? value.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                : value,
            zoom: {
              filterMode: 'discard',
            },
          },
        ]}
        yAxis={[
          {
            id: 'price',
            valueFormatter: formatAsDollar,
            position: 'right',
          },
          ...(hasVolume ? [
            {
              id: 'volume',
              domainLimit: (min, max) => ({ min: 0, max: max.valueOf() * 5 }),
            }
          ] : []),
        ]}
        height={height}
        margin={{ top: 20, bottom: 30, left: 20, right: 60 }}
      >
        <ChartsWrapper sx={{ width: '100%' }}>
          <ChartsLayerContainer>
            <ChartsSvgLayer>
              <ChartsGrid horizontal vertical />
            </ChartsSvgLayer>
            <ChartsWebGLLayer>
              <CandlestickPlot />
            </ChartsWebGLLayer>
            <ChartsSvgLayer>
              <g clipPath={`url(#${clipPathId})`}>
                <BarPlot renderer="svg-batch" />
                <LinePlot />
                <ChartsAxisHighlight x="line" y="line" />
              </g>
              <ChartsClipPath id={clipPathId} />
              <ChartsXAxis />
              <ChartsYAxis axisId="price" />
              {hasVolume && <ChartsYAxis axisId="volume" sx={{ display: 'none' }} />}
              <CandlestickTooltip hasVolume={hasVolume} formatVolume={formatVolume} />
            </ChartsSvgLayer>
          </ChartsLayerContainer>
        </ChartsWrapper>
      </ChartsDataProviderPremium>
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

  React.useEffect(() => {
    if (!plotId) return;
    
    let isMounted = true;
    setLoading(true);
    
    fetch(`http://127.0.0.1:8000/api/plots/${plotId}`)
      .then(res => {
        if (!res.ok) throw new Error('Plot fetch failed');
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
        if (isMounted) setLoading(false);
      });
      
    return () => { isMounted = false; };
  }, [plotId]);

  if (loading) {
    return (
      <Paper elevation={3} sx={{ p: 4, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, display: 'flex', justifyContent: 'center', border: '1px solid #1f2937' }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>Loading visualization...</Typography>
      </Paper>
    );
  }

  if (!spec || !spec.plot_type) return null;

  let ChartComponent;
  switch (spec.plot_type) {
    case 'line':        ChartComponent = SpecLineChart; break;
    case 'bar':         ChartComponent = SpecBarChart;  break;
    case 'pie':         ChartComponent = SpecPieChart;  break;
    case 'scatter':     ChartComponent = SpecScatterChart; break;
    case 'sparkline':   ChartComponent = SpecSparkLineChart; break;
    case 'sankey':      ChartComponent = SpecSankeyChart; break;
    case 'candlestick': ChartComponent = SpecCandlestickChart; break;
    case 'network':     ChartComponent = SpecNetworkChart; break;
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
        overflow: 'hidden',
      }}
    >
      <Typography
        variant="subtitle2"
        align="center"
        sx={{ color: '#e5e7eb', mb: 0.5, fontWeight: 600, letterSpacing: 0.3 }}
      >
        {spec.title}
      </Typography>
      <ChartComponent spec={spec} />
    </Paper>
  );
}
