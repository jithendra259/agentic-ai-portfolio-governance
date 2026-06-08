import React from 'react';
import { Box, Typography, Paper, Skeleton } from '@mui/material';
import { BACKEND_BASE } from '../config/api';
import SmartBarChartRenderer from './charts/bar/SmartBarChartRenderer';
import {
  CandlestickChart,
  FunnelChartRenderer,
  GaugeChartRenderer,
  HeatmapChartRenderer,
  LineChartRenderer,
  NetworkChartRenderer,
  PieChartRenderer,
  RadarChartRenderer,
  RadialBarChartRenderer,
  RadialLineChartRenderer,
  SankeyChartRenderer,
  ScatterChartRenderer,
  SparklineChartRenderer,
} from './charts';

export default function InlineChart({ plotId }) {
  const [spec, setSpec] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    if (!plotId) return;
    
    let isMounted = true;
    setLoading(true);
    setError('');
    
    fetch(`${BACKEND_BASE}/api/plots/${plotId}`)
      .then(async res => {
        if (!res.ok) {
          let detail = '';
          try {
            const payload = await res.json();
            detail = payload?.detail || '';
          } catch {
            detail = await res.text().catch(() => '');
          }
          const message = detail || (res.status === 404 ? 'Plot not found or expired' : 'Plot fetch failed');
          throw new Error(message);
        }
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
        if (isMounted) {
          setError(err.message || 'Visualization unavailable');
          setLoading(false);
        }
      });
      
    return () => { isMounted = false; };
  }, [plotId]);

  if (loading) {
    return (
      <Paper elevation={3} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #1f2937' }}>
        <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
          Loading visualization...
        </Typography>
        <Skeleton variant="rounded" height={24} sx={{ bgcolor: 'rgba(148, 163, 184, 0.12)', mb: 1.5 }} />
        <Skeleton variant="rounded" height={320} sx={{ bgcolor: 'rgba(148, 163, 184, 0.08)' }} />
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper elevation={0} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #374151' }}>
        <Typography variant="body2" sx={{ color: '#fca5a5' }}>
          Visualization unavailable: {error}
        </Typography>
      </Paper>
    );
  }

  if (!spec || !spec.plot_type) return null;

  const displayTitle = (spec.title || '').replace(/\s*\(Interactive Pro\)\s*$/i, ' (Interactive)');

  let ChartComponent;
  switch (spec.plot_type) {
    case 'line':        ChartComponent = LineChartRenderer; break;
    case 'bar':         ChartComponent = SmartBarChartRenderer;  break;
    case 'pie':         ChartComponent = PieChartRenderer;  break;
    case 'scatter':     ChartComponent = ScatterChartRenderer; break;
    case 'sparkline':   ChartComponent = SparklineChartRenderer; break;
    case 'sankey':      ChartComponent = SankeyChartRenderer; break;
    case 'candlestick': ChartComponent = CandlestickChart; break;
    case 'heatmap':     ChartComponent = HeatmapChartRenderer; break;
    case 'network':     ChartComponent = NetworkChartRenderer; break;
    case 'funnel':      ChartComponent = FunnelChartRenderer; break;
    case 'radar':       ChartComponent = RadarChartRenderer; break;
    case 'gauge':       ChartComponent = GaugeChartRenderer; break;
    case 'radial_bar':  ChartComponent = RadialBarChartRenderer; break;
    case 'radial_line': ChartComponent = RadialLineChartRenderer; break;
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
        overflowX: 'auto',
        overflowY: 'hidden',
        maxWidth: '100%',
      }}
    >
      <Typography
        variant="subtitle2"
        align="center"
        sx={{ color: '#e5e7eb', mb: 0.5, fontWeight: 600, letterSpacing: 0.3 }}
      >
        {displayTitle}
      </Typography>
      <ChartComponent spec={spec} />
    </Paper>
  );
}
