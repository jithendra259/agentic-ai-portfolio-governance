import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { BarChartPremium } from '@mui/x-charts-premium/BarChartPremium';
import { adaptBarChartPayload } from './barChartDataAdapter.js';
import { chooseRenderer } from './barChartIntelligence.js';

const AXIS_STYLE = { fill: '#e5e7eb', fontSize: 12, fontWeight: 600 };
const GRID_STYLE = { stroke: '#2b3138', strokeWidth: 1 };

function useResponsiveChartWidth(fallback = 360) {
  const ref = React.useRef(null);
  const [width, setWidth] = React.useState(fallback);

  React.useEffect(() => {
    if (!ref.current) return undefined;
    const updateWidth = (value) => setWidth(Math.max(300, Math.floor(value || fallback)));
    updateWidth(ref.current.getBoundingClientRect().width);
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) updateWidth(entries[0].contentRect.width);
    });
    resizeObserver.observe(ref.current);
    return () => resizeObserver.disconnect();
  }, [fallback]);

  return [ref, width];
}

export default function SmartBarChartRenderer({ spec }) {
  const adapted = useMemo(() => adaptBarChartPayload(spec), [spec]);
  const [chartRef, chartWidth] = useResponsiveChartWidth();

  if (!adapted.valid) {
    return (
      <Box sx={{ p: 2, minHeight: 180, display: 'grid', placeItems: 'center', color: '#9ca3af' }}>
        <Typography variant="body2">{adapted.reason || 'No bar chart data available.'}</Typography>
      </Box>
    );
  }

  const renderer = adapted.renderer || chooseRenderer(adapted.payload);

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <BarChartPremium
        width={chartWidth}
        height={adapted.chartHeight}
        dataset={adapted.dataset}
        series={adapted.series}
        xAxis={decorateAxes(adapted.xAxis)}
        yAxis={decorateAxes(adapted.yAxis)}
        layout={adapted.layout}
        renderer={renderer}
        margin={adapted.margin}
        grid={adapted.grid}
        borderRadius={adapted.borderRadius}
        slotProps={{
          legend: {
            position: { vertical: 'top', horizontal: 'middle' },
            sx: { color: '#e5e7eb', fontSize: 12, fontWeight: 600 },
          },
        }}
        sx={{
          '& .MuiChartsAxis-tickLabel': AXIS_STYLE,
          '& .MuiChartsAxis-label': { fill: '#e5e7eb', fontSize: 12, fontWeight: 700 },
          '& .MuiChartsGrid-line': GRID_STYLE,
          '& .MuiBarLabel-root': { fill: '#fff', fontSize: 12, fontWeight: 800 },
          '& .MuiChartsLegend-label': { color: '#e5e7eb !important' },
        }}
      />
      <BarThresholdNotes thresholds={adapted.thresholds} />
      <BarWarnings warnings={adapted.warnings} interpretation={adapted.interpretation} />
    </Box>
  );
}

function decorateAxes(axes) {
  return (axes || []).map((axis) => ({
    ...axis,
    tickLabelStyle: axis.tickLabelStyle || AXIS_STYLE,
  }));
}

function BarThresholdNotes({ thresholds }) {
  if (!thresholds?.length) return null;
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
      {thresholds.map((threshold) => (
        <Box
          key={`${threshold.name}-${threshold.value}`}
          sx={{
            px: 1,
            py: 0.5,
            border: '1px solid #374151',
            borderRadius: '6px',
            color: '#d1d5db',
            fontSize: 12,
          }}
        >
          {threshold.name}: {threshold.value}
        </Box>
      ))}
    </Box>
  );
}

function BarWarnings({ warnings, interpretation }) {
  const notes = [...(warnings || []), interpretation].filter(Boolean);
  if (!notes.length) return null;
  return (
    <Box sx={{ mt: 1, color: '#9ca3af', fontSize: 12, lineHeight: 1.45 }}>
      {notes.map((note) => (
        <Typography key={note} variant="caption" sx={{ display: 'block', color: 'inherit' }}>
          {note}
        </Typography>
      ))}
    </Box>
  );
}
