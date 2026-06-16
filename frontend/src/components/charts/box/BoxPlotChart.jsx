import React from 'react';
import { Box, Typography } from '@mui/material';

function finite(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function collectDomain(rows) {
  const values = [];
  rows.forEach((row) => {
    ['min', 'q1', 'median', 'q3', 'max'].forEach((key) => {
      const value = finite(row[key]);
      if (value != null) values.push(value);
    });
    (row.outliers || []).forEach((value) => {
      const next = finite(value);
      if (next != null) values.push(next);
    });
  });
  if (!values.length) return [-1, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * 0.12, 0.2);
  return [min - pad, max + pad];
}

export default function BoxPlotChartRenderer({ spec }) {
  const rows = Array.isArray(spec?.data) ? spec.data : [];
  const width = Math.max(520, rows.length * 150 + 120);
  const height = Number(spec?.height) || 380;
  const margin = { top: 24, right: 36, bottom: 62, left: 72 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const [minY, maxY] = collectDomain(rows);
  const yScale = (value) => margin.top + ((maxY - value) / (maxY - minY)) * innerHeight;
  const xStep = innerWidth / Math.max(rows.length, 1);
  const boxWidth = Math.min(70, xStep * 0.45);
  const zeroY = minY <= 0 && maxY >= 0 ? yScale(0) : null;

  if (!rows.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No box plot data available.</Typography>
      </Box>
    );
  }

  const ticks = Array.from({ length: 5 }, (_, index) => minY + ((maxY - minY) * index) / 4);

  return (
    <Box sx={{ width: '100%', overflowX: 'auto', py: 1 }}>
      <svg width={width} height={height} role="img" aria-label={spec.title || 'Box plot'} style={{ display: 'block' }}>
        <rect x="0" y="0" width={width} height={height} fill="#111827" />
        {ticks.map((tick) => {
          const y = yScale(tick);
          return (
            <g key={tick}>
              <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} stroke="rgba(148,163,184,0.18)" />
              <text x={margin.left - 10} y={y + 4} textAnchor="end" fill="#9ca3af" fontSize="11">
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}
        {zeroY != null && (
          <line x1={margin.left} x2={width - margin.right} y1={zeroY} y2={zeroY} stroke="rgba(229,231,235,0.45)" strokeDasharray="4 4" />
        )}
        <text
          x={18}
          y={margin.top + innerHeight / 2}
          fill="#cbd5e1"
          fontSize="12"
          fontWeight="700"
          transform={`rotate(-90 18 ${margin.top + innerHeight / 2})`}
          textAnchor="middle"
        >
          {spec.y_label || 'Value'}
        </text>

        {rows.map((row, index) => {
          const centerX = margin.left + xStep * index + xStep / 2;
          const yMin = yScale(row.min);
          const yQ1 = yScale(row.q1);
          const yMedian = yScale(row.median);
          const yQ3 = yScale(row.q3);
          const yMax = yScale(row.max);
          const outliers = (row.outliers || []).slice(0, 40);
          return (
            <g key={row.label || index}>
              <line x1={centerX} x2={centerX} y1={yMax} y2={yMin} stroke="#93c5fd" strokeWidth="2" />
              <line x1={centerX - boxWidth / 3} x2={centerX + boxWidth / 3} y1={yMin} y2={yMin} stroke="#93c5fd" strokeWidth="2" />
              <line x1={centerX - boxWidth / 3} x2={centerX + boxWidth / 3} y1={yMax} y2={yMax} stroke="#93c5fd" strokeWidth="2" />
              <rect
                x={centerX - boxWidth / 2}
                y={Math.min(yQ1, yQ3)}
                width={boxWidth}
                height={Math.max(2, Math.abs(yQ1 - yQ3))}
                rx="4"
                fill="rgba(59,130,246,0.36)"
                stroke="#60a5fa"
                strokeWidth="2"
              />
              <line x1={centerX - boxWidth / 2} x2={centerX + boxWidth / 2} y1={yMedian} y2={yMedian} stroke="#fbbf24" strokeWidth="3" />
              {outliers.map((value, outlierIndex) => {
                const jitter = ((outlierIndex % 7) - 3) * 2.4;
                return (
                  <circle
                    key={`${row.label}-outlier-${outlierIndex}`}
                    cx={centerX + jitter}
                    cy={yScale(value)}
                    r="2.5"
                    fill="#f87171"
                    opacity="0.82"
                  />
                );
              })}
              <text x={centerX} y={height - margin.bottom + 24} textAnchor="middle" fill="#e5e7eb" fontSize="12" fontWeight="700">
                {row.label}
              </text>
              <text x={centerX} y={height - margin.bottom + 42} textAnchor="middle" fill="#9ca3af" fontSize="10">
                n={row.sample_size}
              </text>
            </g>
          );
        })}
      </svg>
    </Box>
  );
}
