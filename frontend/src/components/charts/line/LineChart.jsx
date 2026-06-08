import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { useTheme, alpha } from '@mui/material/styles';
import { useDrawingArea, useXScale } from '@mui/x-charts/hooks';

const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
const AXIS_STYLE = { fill: '#9ca3af', fontSize: 11 };
const GRID_STYLE = { stroke: '#374151', strokeDasharray: '4 4' };

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getValueFormatter(format) {
  if (format === 'percent' || format === '%') return (v) => (v == null ? '' : `${v.toFixed(1)}%`);
  if (format === 'decimal') return (v) => (v == null ? '' : Number(v).toFixed(2));
  if (format === 'beta') return (v) => (v == null ? '' : `${Number(v).toFixed(2)} beta`);
  if (format === 'k') return (v) => (v == null ? '' : `${(v / 1000).toFixed(1)}k`);
  if (format === 'currency') return (v) => (v == null ? '' : `$${v.toLocaleString()}`);
  return (v) => {
    if (v == null) return '';
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return Number(v).toFixed(2);
  };
}

function useResponsiveChartWidth(fallback = 360) {
  const ref = React.useRef(null);
  const [width, setWidth] = React.useState(fallback);

  React.useEffect(() => {
    if (!ref.current) return;
    const updateWidth = (value) => setWidth(Math.max(280, Math.floor(value || fallback)));
    updateWidth(ref.current.getBoundingClientRect().width);
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) updateWidth(entries[0].contentRect.width);
    });
    resizeObserver.observe(ref.current);
    return () => resizeObserver.disconnect();
  }, [fallback]);

  return [ref, width];
}

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
        if (xStart === undefined || xEnd === undefined || isNaN(xStart) || isNaN(xEnd)) return null;
        let startX = xStart;
        let endX = xEnd;
        if (startX < left) startX = left;
        if (endX > left + width) endX = left + width;
        if (startX >= endX) return null;
        const textX = xStart >= left ? xStart : left;
        return (
          <React.Fragment key={index}>
            <rect x={startX} y={top} width={endX - startX} height={height} fill="grey" opacity={0.15} />
            <text x={textX + 4} y={top - 5} textAnchor="start" dominantBaseline="auto" fill={labelFill} fontSize="0.75rem" fontWeight={500} pointerEvents="none">
              {p.label}
            </text>
          </React.Fragment>
        );
      })}
    </g>
  );
}

export default function LineChartRenderer({ spec }) {
  const { dataset, series, yAxisConfig, margins } = useMemo(() => {
    if (!spec?.series?.length) return { dataset: [], series: [], yAxisConfig: [], margins: { top: 24, right: 24, left: 60, bottom: 40 } };
    const dateSet = new Set();
    spec.series.forEach((s) => s.data?.forEach((pt) => dateSet.add(pt.x)));
    const sortedDates = Array.from(dateSet).sort();
    const byDate = {};
    sortedDates.forEach((d) => { byDate[d] = { date: new Date(d) }; });
    spec.series.forEach((s) => {
      s.data?.forEach((pt) => {
        if (byDate[pt.x]) byDate[pt.x][s.name] = pt.y;
      });
    });
    const dataset = sortedDates.map((d) => byDate[d]);
    const series = spec.series.map((s, i) => {
      const entry = {
        type: 'line',
        dataKey: s.name,
        label: s.label || s.name,
        color: s.color || PALETTE[i % PALETTE.length],
        valueFormatter: getValueFormatter(s.value_format || spec.y_format),
      };
      if (s.yAxisId) entry.yAxisId = s.yAxisId;
      if (s.area != null) entry.area = s.area;
      if (s.baseline != null) entry.baseline = s.baseline;
      if (s.stack) entry.stack = s.stack;
      if (s.stackOffset) entry.stackOffset = s.stackOffset;
      entry.showMark = s.showMark ?? false;
      if (s.shape) entry.shape = s.shape;
      entry.connectNulls = s.connectNulls ?? spec.connect_nulls ?? spec.connectNulls ?? false;
      if (s.highlightScope) entry.highlightScope = s.highlightScope;
      else if (spec.highlightScope) entry.highlightScope = spec.highlightScope;
      if (s.disableHighlight != null) entry.disableHighlight = s.disableHighlight;
      if (s.curve) entry.curve = s.curve;
      else if (spec.curve) entry.curve = spec.curve;
      return entry;
    });
    let yAxisConfig = [];
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      yAxisConfig = spec.yAxis.map((axis) => ({
        id: axis.id,
        label: axis.label || '',
        position: axis.position || 'left',
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(axis.value_format),
        width: axis.width || (axis.position === 'right' ? 50 : 55),
        domainLimit: axis.domainLimit || 'nice',
        ...(axis.colorMap ? { colorMap: axis.colorMap } : {}),
      }));
    } else {
      yAxisConfig = [{ id: 'default-y-axis', tickLabelStyle: AXIS_STYLE, label: spec.y_label || '', valueFormatter: getValueFormatter(spec.y_format), domainLimit: 'nice' }];
    }
    const hasRightAxis = spec.yAxis?.some((axis) => axis.position === 'right');
    const margins = { top: 60, right: hasRightAxis ? 60 : 24, left: 60, bottom: 60 };
    return { dataset, series, yAxisConfig, margins };
  }, [spec]);

  if (!dataset.length) return null;
  const gridConfig = spec.grid || { horizontal: true };
  const animationSx = useMemo(() => (spec.animation ? {
    '& .MuiLineElement-root.MuiCharts-animate': { animationDuration: spec.animation.duration || '1s', animationDelay: spec.animation.delay || '0s', animationTimingFunction: spec.animation.easing || 'ease-out' },
    '& .MuiAreaElement-root.MuiCharts-animate': { animationDuration: spec.animation.duration || '1s', animationDelay: spec.animation.delay || '0s', animationTimingFunction: spec.animation.easing || 'ease-out' },
    '& .MuiChartsGrid-line': GRID_STYLE,
  } : {}), [spec.animation]);
  const [chartRef, chartWidth] = useResponsiveChartWidth();
  return (
    <Box ref={chartRef} sx={{ width: '100%', height: 320, minWidth: 0, ...animationSx }}>
      <LineChart
        width={chartWidth}
        dataset={dataset}
        series={series}
        xAxis={[{ id: 'x-axis', dataKey: 'date', scaleType: 'time', tickLabelStyle: AXIS_STYLE, label: spec.x_label || 'Date' }]}
        yAxis={yAxisConfig}
        margin={margins}
        grid={gridConfig}
        {...(spec.skipAnimation ? { skipAnimation: true } : {})}
      >
        {spec.recessions && <RecessionBands periods={spec.recessions} />}
      </LineChart>
    </Box>
  );
}
