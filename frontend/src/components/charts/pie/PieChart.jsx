import React, { useMemo } from 'react';
import { Box } from '@mui/material';
import { PieChart } from '@mui/x-charts/PieChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import { PALETTE, getPieShare } from './pieChartUtils';
import PieCenterLabel from './PieCenterLabel';

export default function PieChartRenderer({ spec }) {
  const mappedSeries = useMemo(() => {
    if (!spec?.series || !Array.isArray(spec.series)) return [];
    return spec.series.map((s) => {
      const rawData = s.data || [];
      const total = rawData.reduce((sum, pt) => sum + (pt.value != null ? pt.value : (pt.y || 0)), 0);
      const maxVal = rawData.reduce((max, pt) => Math.max(max, pt.value != null ? pt.value : (pt.y || 0)), 0);
      const isFractional = maxVal <= 1.0;
      
      let dataPoints = rawData.map((pt, i) => {
        const id = pt.id || pt.x || `slice-${i}`;
        const value = pt.value != null ? pt.value : (pt.y != null ? pt.y : 0);
        const color = pt.color || PALETTE[i % PALETTE.length];
        const formatVal = (v) => {
          if (s.valueFormatter === 'percent' || (!s.valueFormatter && isFractional)) return `${(v * (isFractional ? 100 : 1)).toFixed(1)}%`;
          if (s.valueFormatter === 'currency') return `$${v.toLocaleString()}`;
          if (s.valueFormatter === 'raw') return v.toString();
          return `${getPieShare(v, total).toFixed(1)}%`;
        };
        const entry = { id, value, color, label: pt.label != null ? pt.label : `${id} (${formatVal(value)})` };
        if (pt.labelMarkType) entry.labelMarkType = pt.labelMarkType;
        return entry;
      });

      if (s.sorting === 'asc') dataPoints = [...dataPoints].sort((a, b) => a.value - b.value);
      else if (s.sorting === 'desc') dataPoints = [...dataPoints].sort((a, b) => b.value - a.value);

      const entry = {
        data: dataPoints,
        innerRadius: s.innerRadius !== undefined ? s.innerRadius : 48,
        outerRadius: s.outerRadius !== undefined ? s.outerRadius : 110,
        paddingAngle: s.paddingAngle !== undefined ? s.paddingAngle : 2,
        cornerRadius: s.cornerRadius !== undefined ? s.cornerRadius : 4,
      };

      if (s.startAngle !== undefined) entry.startAngle = s.startAngle;
      if (s.endAngle !== undefined) entry.endAngle = s.endAngle;
      if (s.cx !== undefined) entry.cx = s.cx;
      if (s.cy !== undefined) entry.cy = s.cy;
      if (s.arcLabelRadius !== undefined) entry.arcLabelRadius = s.arcLabelRadius;
      if (s.arcLabelMinAngle !== undefined) entry.arcLabelMinAngle = s.arcLabelMinAngle;
      
      if (s.arcLabel) {
        if (typeof s.arcLabel === 'string') {
          if (s.arcLabel === 'percent') entry.arcLabel = (item) => `${(item.value / total * 100).toFixed(0)}%`;
          else if (s.arcLabel === 'label-percent') entry.arcLabel = (item) => `${(dataPoints.find((d) => d.id === item.id)?.label || item.id).split(' (')[0]} (${(item.value / total * 100).toFixed(0)}%)`;
          else entry.arcLabel = s.arcLabel;
        } else entry.arcLabel = s.arcLabel;
      }

      entry.valueFormatter = (item) => {
        const val = item.value;
        if (s.valueFormatter === 'percent' || (!s.valueFormatter && isFractional)) return `${(val * (isFractional ? 100 : 1)).toFixed(1)}%`;
        if (s.valueFormatter === 'currency') return `$${val.toLocaleString()}`;
        if (s.valueFormatter === 'raw') return val.toString();
        return `${getPieShare(val, total).toFixed(1)}%`;
      };

      if (s.highlightScope) entry.highlightScope = s.highlightScope;
      else if (spec.highlightScope) entry.highlightScope = spec.highlightScope;
      if (s.faded) entry.faded = s.faded;
      if (s.highlighted) entry.highlighted = s.highlighted;
      return entry;
    });
  }, [spec]);

  const [chartRef, chartWidth] = useResponsiveChartWidth(360, 280);
  const compact = chartWidth < 460;
  const chartHeight = compact ? Math.min(320, spec.height || 320) : (spec.height || 320);
  const centerLabel = spec.centerLabel || spec.center_label;
  
  const responsiveSeries = useMemo(() => 
    compact 
      ? mappedSeries.map((series) => ({ 
          ...series, 
          innerRadius: Math.min(series.innerRadius ?? 0, 48), 
          outerRadius: Math.min(series.outerRadius ?? 110, 86) 
        })) 
      : mappedSeries, 
    [compact, mappedSeries]
  );

  if (!mappedSeries.length) return null;
  const legendItems = spec.hideLegend ? [] : responsiveSeries.flatMap((series) => series.data || []);

  return (
    <Box ref={chartRef} sx={{ width: '100%', minWidth: 0 }}>
      <PieChart
        width={chartWidth}
        series={responsiveSeries}
        height={chartHeight}
        margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
        hideLegend
        {...(spec.colors && Array.isArray(spec.colors) ? { colors: spec.colors } : {})}
      >
        {centerLabel && <PieCenterLabel>{centerLabel}</PieCenterLabel>}
      </PieChart>
      {legendItems.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 1.25, mt: -1, px: 1, color: '#e5e7eb', fontSize: 11, lineHeight: 1.2 }}>
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
