import React from 'react';
import { Box } from '@mui/material';
import { Gauge } from '@mui/x-charts-premium/Gauge';

const PALETTE = ['#3b82f6'];
function toFiniteNumber(value, fallback = 0) { const next = Number(value); return Number.isFinite(next) ? next : fallback; }
function getResponsiveChartHeight(spec, fallback = 260) { const requested = Number(spec?.height); return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback; }

export default function GaugeChartRenderer({ spec }) {
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
