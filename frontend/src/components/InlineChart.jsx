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
  BoxPlotChartRenderer,
} from './charts';

function useNearViewport(rootMargin = '600px') {
  const ref = React.useRef(null);
  const [isNearViewport, setIsNearViewport] = React.useState(() => typeof IntersectionObserver === 'undefined');

  React.useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === 'undefined') {
      setIsNearViewport(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => setIsNearViewport(entry.isIntersecting),
      { root: null, rootMargin, threshold: 0.01 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [rootMargin]);

  return [ref, isNearViewport];
}

function getPointCount(spec) {
  if (Number.isFinite(Number(spec?.density?.rendered_points))) {
    return Number(spec.density.rendered_points);
  }
  return (spec?.series || []).reduce((total, entry) => total + (Array.isArray(entry?.data) ? entry.data.length : 0), 0);
}

class ChartErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error('Chart render failed:', error);
  }

  render() {
    if (this.state.error) {
      return (
        <Box sx={{ p: 2, border: '1px solid #374151', borderRadius: 1, bgcolor: '#0b1120' }}>
          <Typography variant="body2" sx={{ color: '#fca5a5' }}>
            Visualization unavailable: chart renderer could not display this PlotSpec.
          </Typography>
        </Box>
      );
    }

    return this.props.children;
  }
}

function InlineChart({ plotId }) {
  const [containerRef, isNearViewport] = useNearViewport();
  const [spec, setSpec] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    setSpec(null);
    setLoading(true);
    setError('');
  }, [plotId]);

  React.useEffect(() => {
    if (!plotId || !isNearViewport || spec) return;
    
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
  }, [plotId, isNearViewport, spec]);

  const placeholder = (
    <Paper
      ref={containerRef}
      elevation={0}
      sx={{
        p: 2,
        mt: 1,
        mb: 1,
        width: '100%',
        minHeight: 180,
        bgcolor: '#111827',
        borderRadius: 2,
        border: '1px solid #1f2937',
        contentVisibility: 'auto',
        containIntrinsicSize: '180px',
      }}
    >
      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
        Visualization will load when visible.
      </Typography>
    </Paper>
  );

  if (!isNearViewport && !spec) {
    return placeholder;
  }

  if (loading) {
    return (
      <Paper ref={containerRef} elevation={3} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #1f2937' }}>
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
      <Paper ref={containerRef} elevation={0} sx={{ p: 2, mt: 1, mb: 1, width: '100%', bgcolor: '#111827', borderRadius: 2, border: '1px solid #374151' }}>
        <Typography variant="body2" sx={{ color: '#fca5a5' }}>
          Visualization unavailable: {error}
        </Typography>
      </Paper>
    );
  }

  if (!spec || !spec.plot_type) return null;
  if (!isNearViewport) {
    return (
      <Paper
        ref={containerRef}
        elevation={0}
        sx={{
          p: 2,
          mt: 1,
          mb: 1,
          width: '100%',
          minHeight: 180,
          bgcolor: '#111827',
          borderRadius: 2,
          border: '1px solid #1f2937',
          contentVisibility: 'auto',
          containIntrinsicSize: '180px',
        }}
      >
        <Typography variant="subtitle2" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
          {(spec.title || 'Visualization').replace(/\s*\(Interactive Pro\)\s*$/i, ' (Interactive)')}
        </Typography>
        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
          Chart paused offscreen to keep chat responsive.
        </Typography>
      </Paper>
    );
  }

  const displayTitle = (spec.title || '').replace(/\s*\(Interactive Pro\)\s*$/i, ' (Interactive)');
  const pointCount = getPointCount(spec);

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
    case 'box':         ChartComponent = BoxPlotChartRenderer; break;
    default:            return null;
  }

  return (
    <Paper
      ref={containerRef}
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
        contentVisibility: 'auto',
        containIntrinsicSize: '380px',
      }}
    >
      <Typography
        variant="subtitle2"
        align="center"
        sx={{ color: '#e5e7eb', mb: 0.5, fontWeight: 600, letterSpacing: 0.3 }}
      >
        {displayTitle}
      </Typography>
      {spec.density?.sampled && (
        <Typography variant="caption" align="center" sx={{ color: '#94a3b8', display: 'block', mb: 0.5 }}>
          Rendered {pointCount.toLocaleString()} sampled points for smooth chat scrolling.
        </Typography>
      )}
      <ChartErrorBoundary key={plotId}>
        <ChartComponent spec={spec} />
      </ChartErrorBoundary>
    </Paper>
  );
}

export default React.memo(InlineChart);
