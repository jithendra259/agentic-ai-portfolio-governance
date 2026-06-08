import React from 'react';
import { Box } from '@mui/material';
import { Gauge } from '@mui/x-charts-premium/Gauge';
import { PALETTE, toFiniteNumber, getResponsiveChartHeight } from './gaugeChartUtils';

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
