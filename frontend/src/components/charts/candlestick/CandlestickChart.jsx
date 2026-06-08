import React, { useMemo, useState } from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import {
  getResponsiveChartHeight,
  formatVolume,
  formatAsDollar,
  layoutCandlestick,
} from './candlestickChartUtils';

export default function CandlestickChart({ spec }) {
  const height = getResponsiveChartHeight(spec, 320);
  const [containerRef, containerWidth] = useResponsiveChartWidth(360, 320);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  
  const primarySeries = Array.isArray(spec?.series)
    ? spec.series.find((series) => Array.isArray(series?.data) && series.data.length > 0)
    : null;
  const pts = primarySeries?.data || (Array.isArray(spec?.data) ? spec.data : []);

  const chart = useMemo(() => {
    return layoutCandlestick({ pts, containerWidth, height });
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
            sx={{ height: 22, borderColor: '#334155', color: '#cbd5e1', bgcolor: 'rgba(15, 23, 42, 0.55)', '& .MuiChip-label': { px: 1, fontSize: 11, fontWeight: 600 } }}
          />
        )}
      </Box>
      <svg width="100%" height={height} viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label={spec.title || 'Candlestick chart'}>
        <rect x="0" y="0" width={chart.width} height={chart.height} fill="transparent" />
        <g>
          {chart.priceTicks.map((tick) => (
            <g key={tick.value}>
              <line x1={chart.margin.left} x2={chart.margin.left + chart.innerWidth} y1={tick.y} y2={tick.y} stroke="#374151" strokeDasharray="4 4" />
              <text x={chart.margin.left + chart.innerWidth + 8} y={tick.y + 4} fill="#cbd5e1" fontSize="11" fontWeight="700">
                {formatAsDollar(tick.value)}
              </text>
            </g>
          ))}
          {chart.dateTicks.map((tick) => (
            <g key={`${tick.x}-${tick.label}`}>
              <line x1={tick.x} x2={tick.x} y1={chart.margin.top} y2={chart.hasVolume ? chart.plotBottom + chart.volumeGap + chart.volumeHeight : chart.plotBottom} stroke="#334155" strokeOpacity="0.55" />
              <text x={tick.x} y={height - 14} fill="#cbd5e1" fontSize="11" textAnchor="middle">
                {tick.label}
              </text>
            </g>
          ))}
        </g>
        {chart.hasVolume && (
          <g>
            <line x1={chart.margin.left} x2={chart.margin.left + chart.innerWidth} y1={chart.plotBottom + chart.volumeGap + chart.volumeHeight} y2={chart.plotBottom + chart.volumeGap + chart.volumeHeight} stroke="#475569" />
            {chart.data.map((entry) => {
              const rising = entry.index === 0 || entry.close >= chart.data[entry.index - 1].close;
              const x = chart.xFor(entry.index) - chart.candleWidth / 2;
              const y = chart.volumeY(entry.volume);
              const h = chart.plotBottom + chart.volumeGap + chart.volumeHeight - y;
              return <rect key={`volume-${entry.index}`} x={x} y={y} width={chart.candleWidth} height={Math.max(1, h)} fill={rising ? '#22c55e' : '#ef4444'} opacity="0.78" />;
            })}
          </g>
        )}
        <g>
          {chart.movingAveragePath && <path d={chart.movingAveragePath} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" />}
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
              <g key={`candle-${entry.index}`} pointerEvents="none">
                <line x1={x} x2={x} y1={highY} y2={lowY} stroke={color} strokeWidth="1.5" />
                <rect x={x - chart.candleWidth / 2} y={bodyY} width={chart.candleWidth} height={bodyHeight} rx="1" fill={color} />
                <rect x={x - chart.candleWidth / 2 - 3} y={chart.margin.top} width={chart.candleWidth + 6} height={chart.priceHeight + chart.volumeGap + chart.volumeHeight} fill="transparent">
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
        <Box sx={{ position: 'absolute', top: 12, left: tooltipLeft, pointerEvents: 'none', background: 'rgba(15, 23, 42, 0.92)', border: '1px solid #334155', borderRadius: 1, color: '#e5e7eb', px: 1, py: 0.75, fontSize: 11, boxShadow: '0 10px 24px rgba(0,0,0,0.25)', zIndex: 2 }}>
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
