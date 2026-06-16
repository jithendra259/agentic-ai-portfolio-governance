import React from 'react';
import { Box, Typography } from '@mui/material';
import { toFiniteNumber } from './heatmapChartUtils';

function axisLabels(axis, fallback = []) {
  const data = Array.isArray(axis) ? axis[0]?.data : axis?.data;
  return Array.isArray(data) ? data.map(String) : fallback.map(String);
}

function buildMatrix(spec) {
  const matrixObject = spec.matrix || spec.correlation_matrix || spec.covariance_matrix;
  if (matrixObject && typeof matrixObject === 'object' && !Array.isArray(matrixObject)) {
    const yLabels = Object.keys(matrixObject);
    const xLabels = Object.keys(matrixObject[yLabels[0]] || {});
    return {
      xLabels,
      yLabels,
      values: yLabels.map((row) => xLabels.map((col) => toFiniteNumber(matrixObject[row]?.[col], null))),
    };
  }

  const xLabels = axisLabels(spec.xAxis, spec.xLabels || []);
  const yLabels = axisLabels(spec.yAxis, spec.yLabels || []);
  const values = yLabels.map(() => xLabels.map(() => null));
  const points = spec.series?.[0]?.data || [];

  points.forEach((point) => {
    if (!Array.isArray(point) || point.length < 3) return;
    const xIndex = Number(point[0]);
    const yIndex = Number(point[1]);
    if (!Number.isInteger(xIndex) || !Number.isInteger(yIndex)) return;
    if (!values[yIndex] || values[yIndex][xIndex] === undefined) return;
    values[yIndex][xIndex] = toFiniteNumber(point[2], null);
  });

  return { xLabels, yLabels, values };
}

function heatmapType(spec) {
  return String(spec.metadata?.heatmap_type || spec.heatmap_type || '').toLowerCase();
}

function valueRange(values, type) {
  if (type.includes('correlation')) return { min: -1, max: 1, center: 0 };
  const flat = values.flat().filter((value) => Number.isFinite(value));
  if (!flat.length) return { min: 0, max: 1, center: 0 };
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  return { min, max, center: min < 0 && max > 0 ? 0 : min };
}

function colorForValue(value, range, type) {
  if (!Number.isFinite(value)) return '#111827';

  if (type.includes('correlation')) {
    const magnitude = Math.min(1, Math.abs(value));
    if (value >= 0) {
      return `rgba(34, 197, 94, ${0.16 + magnitude * 0.76})`;
    }
    return `rgba(59, 130, 246, ${0.16 + magnitude * 0.76})`;
  }

  const span = Math.max(1e-9, range.max - range.min);
  const t = Math.max(0, Math.min(1, (value - range.min) / span));
  const hue = 210 - t * 170;
  return `hsl(${hue}, 72%, ${24 + t * 34}%)`;
}

function formatValue(value, type) {
  if (!Number.isFinite(value)) return '-';
  if (type.includes('correlation')) return value.toFixed(4);
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10) return value.toFixed(2);
  return value.toFixed(4);
}

export default function HeatmapChartRenderer({ spec }) {
  const { xLabels, yLabels, values } = React.useMemo(() => buildMatrix(spec || {}), [spec]);
  const type = heatmapType(spec || {});
  const range = React.useMemo(() => valueRange(values, type), [values, type]);

  if (!xLabels.length || !yLabels.length || !values.some((row) => row.some(Number.isFinite))) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No heatmap data available.</Typography>
      </Box>
    );
  }

  const minCellWidth = xLabels.length <= 6 ? 110 : 72;

  return (
    <Box sx={{ width: '100%', overflowX: 'auto', py: 1 }}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: `96px repeat(${xLabels.length}, minmax(${minCellWidth}px, 1fr))`,
          minWidth: 96 + xLabels.length * minCellWidth,
          gap: '1px',
          bgcolor: 'rgba(148, 163, 184, 0.18)',
          border: '1px solid rgba(148, 163, 184, 0.22)',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        <Box sx={{ bgcolor: '#0b1220', minHeight: 42 }} />
        {xLabels.map((label) => (
          <Box key={`x-${label}`} sx={{ bgcolor: '#111827', px: 1, py: 1.25, textAlign: 'center' }}>
            <Typography variant="caption" sx={{ color: '#e5e7eb', fontWeight: 700 }}>
              {label}
            </Typography>
          </Box>
        ))}

        {yLabels.map((rowLabel, rowIndex) => (
          <React.Fragment key={`row-${rowLabel}`}>
            <Box sx={{ bgcolor: '#111827', px: 1.25, py: 1.5, display: 'flex', alignItems: 'center' }}>
              <Typography variant="caption" sx={{ color: '#e5e7eb', fontWeight: 700 }}>
                {rowLabel}
              </Typography>
            </Box>
            {xLabels.map((colLabel, colIndex) => {
              const value = values[rowIndex]?.[colIndex];
              const bg = colorForValue(value, range, type);
              const strong = Number.isFinite(value) && Math.abs(value) > 0.65;
              return (
                <Box
                  key={`${rowLabel}-${colLabel}`}
                  title={`${rowLabel} vs ${colLabel}: ${formatValue(value, type)}`}
                  sx={{
                    minHeight: 72,
                    bgcolor: bg,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: strong ? '#ffffff' : '#e5e7eb',
                    transition: 'transform 120ms ease, filter 120ms ease',
                    '&:hover': {
                      transform: 'scale(1.035)',
                      filter: 'brightness(1.12)',
                      zIndex: 1,
                    },
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
                    {formatValue(value, type)}
                  </Typography>
                </Box>
              );
            })}
          </React.Fragment>
        ))}
      </Box>

      {type.includes('correlation') && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mt: 1.25, px: 0.5 }}>
          <Typography variant="caption" sx={{ color: '#93c5fd' }}>-1</Typography>
          <Box
            sx={{
              height: 10,
              flex: 1,
              borderRadius: 999,
              background: 'linear-gradient(90deg, rgba(59,130,246,.92), rgba(17,24,39,.95), rgba(34,197,94,.92))',
              border: '1px solid rgba(148, 163, 184, 0.24)',
            }}
          />
          <Typography variant="caption" sx={{ color: '#86efac' }}>+1</Typography>
        </Box>
      )}
    </Box>
  );
}
