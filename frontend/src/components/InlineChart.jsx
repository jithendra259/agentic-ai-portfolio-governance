import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import { PieChart } from '@mui/x-charts/PieChart';
import { Box, Typography, Paper } from '@mui/material';
import { useDrawingArea, useXScale } from '@mui/x-charts/hooks';
import { useTheme, alpha } from '@mui/material/styles';

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
    const series = spec.series.map((s, i) => ({
      type: 'line',
      dataKey: s.name,
      label: s.label || s.name,
      color: s.color || PALETTE[i % PALETTE.length],
      showMark: false,
      connectNulls: true,
      yAxisId: s.yAxisId || undefined,
      valueFormatter: getValueFormatter(s.value_format || spec.y_format),
    }));

    let yAxisConfig = [];
    if (spec.yAxis && Array.isArray(spec.yAxis)) {
      yAxisConfig = spec.yAxis.map(axis => ({
        id: axis.id,
        label: axis.label || '',
        position: axis.position || 'left',
        tickLabelStyle: AXIS_STYLE,
        valueFormatter: getValueFormatter(axis.value_format),
        width: axis.width || (axis.position === 'right' ? 50 : 55),
      }));
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

  return (
    <Box sx={{ width: '100%', height: CHART_HEIGHT }}>
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
        }}
      >
        {spec.recessions && <RecessionBands periods={spec.recessions} />}
      </LineChart>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bar chart — spec.series[0].data = [{x: string, y: number}]
// ─────────────────────────────────────────────────────────────────────────────
function SpecBarChart({ spec }) {
  const { dataset, series } = useMemo(() => {
    if (!spec?.series?.length) return { dataset: [], series: [] };

    // Collect all unique categories across all series
    const catSet = new Set();
    spec.series.forEach(s => s.data?.forEach(pt => catSet.add(pt.x)));
    const categories = Array.from(catSet);

    // Build pivoted dataset [{label: "category", AAPL: 10, MSFT: 15}]
    const byCategory = {};
    categories.forEach(c => { byCategory[c] = { label: c }; });
    spec.series.forEach(s => {
      s.data?.forEach(pt => {
        if (byCategory[pt.x]) byCategory[pt.x][s.name] = pt.y;
      });
    });

    const dataset = categories.map(c => byCategory[c]);
    const series = spec.series.map((s, i) => ({
      dataKey: s.name,
      label: s.name,
      color: s.color || PALETTE[i % PALETTE.length],
      valueFormatter: getValueFormatter(s.value_format || 'none'),
    }));

    return { dataset, series };
  }, [spec]);

  if (!dataset.length) return null;

  return (
    <BarChart
      dataset={dataset}
      xAxis={[{
        dataKey: 'label',
        scaleType: 'band',
        tickLabelStyle: { ...AXIS_STYLE, angle: -30, textAnchor: 'end' },
        label: spec.x_label || '',
      }]}
      yAxis={[{
        tickLabelStyle: AXIS_STYLE,
        label: spec.y_label || '',
        valueFormatter: getValueFormatter(spec.y_format),
        domainLimit: 'nice',
      }]}
      series={series}
      height={CHART_HEIGHT}
      margin={{ top: 16, right: 24, left: 60, bottom: 56 }}
      grid={{ horizontal: true }}
      sx={{ '& .MuiChartsGrid-line': GRID_STYLE }}
      slotProps={{ legend: { labelStyle: { fill: '#e5e7eb', fontSize: 12 } } }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pie chart — spec.series[0].data = [{x: label, y: value, color?: string}]
// ─────────────────────────────────────────────────────────────────────────────
function SpecPieChart({ spec }) {
  const pieData = useMemo(() => {
    if (!spec?.series?.length) return [];
    return (spec.series[0]?.data || []).map((pt, i) => ({
      id: pt.x,
      value: pt.y,
      label: `${pt.x} (${(pt.y * 100).toFixed(1)}%)`,
      color: pt.color || PALETTE[i % PALETTE.length],
    }));
  }, [spec]);

  if (!pieData.length) return null;

  return (
    <PieChart
      series={[{
        data: pieData,
        innerRadius: 48,
        outerRadius: 110,
        paddingAngle: 2,
        cornerRadius: 4,
        valueFormatter: item => `${(item.value * 100).toFixed(1)}%`,
      }]}
      height={CHART_HEIGHT}
      margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
      slotProps={{
        legend: {
          labelStyle: { fill: '#e5e7eb', fontSize: 11 },
          itemMarkWidth: 10,
          itemMarkHeight: 10,
          markGap: 6,
          itemGap: 10,
        },
      }}
    />
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
    case 'line':    ChartComponent = SpecLineChart; break;
    case 'bar':     ChartComponent = SpecBarChart;  break;
    case 'pie':     ChartComponent = SpecPieChart;  break;
    default:        return null;
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
