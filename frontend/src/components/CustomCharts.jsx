import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Chip } from '@mui/material';

// ---------------------------------------------------------------------------
// HeatmapChart
// Renders correlation/covariance matrices in a beautiful responsive grid
// ---------------------------------------------------------------------------
export function HeatmapChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return <Typography variant="body2" color="text.secondary">No data available for heatmap</Typography>;
  }

  // Extract unique X and Y tickers
  const tickers = Array.from(new Set(data.map(item => item.tickerX))).sort();

  // Create lookup matrix
  const matrix = {};
  data.forEach(item => {
    if (!matrix[item.tickerX]) matrix[item.tickerX] = {};
    matrix[item.tickerX][item.tickerY] = item.correlation !== undefined ? item.correlation : item.covariance;
  });

  const isCorrelation = data[0].correlation !== undefined;

  const getColor = (val) => {
    if (isCorrelation) {
      // Correlation range: -1 to 1
      // Blue for negative, white for zero, red for positive
      if (val >= 0) {
        const intensity = Math.min(val, 1.0);
        return `rgba(239, 68, 68, ${intensity})`; // Red
      } else {
        const intensity = Math.min(Math.abs(val), 1.0);
        return `rgba(59, 130, 246, ${intensity})`; // Blue
      }
    } else {
      // Covariance: always positive or scaled
      const maxVal = Math.max(...data.map(item => Math.abs(item.covariance || 0))) || 1;
      const intensity = Math.min(Math.abs(val) / maxVal, 1.0);
      return `rgba(16, 185, 129, ${intensity})`; // Green
    }
  };

  return (
    <Box sx={{ overflowX: 'auto', width: '100%', py: 1 }}>
      <TableContainer component={Box} sx={{ background: 'transparent', boxShadow: 'none' }}>
        <Table size="small" sx={{ borderCollapse: 'separate', borderSpacing: '4px' }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ border: 'none', fontWeight: 600, color: 'text.secondary', width: '60px' }}></TableCell>
              {tickers.map(t => (
                <TableCell key={t} align="center" sx={{ border: 'none', fontWeight: 700, color: 'text.primary', fontSize: '0.78rem' }}>
                  {t}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {tickers.map(tx => (
              <TableRow key={tx}>
                <TableCell sx={{ border: 'none', fontWeight: 700, color: 'text.primary', fontSize: '0.78rem' }}>
                  {tx}
                </TableCell>
                {tickers.map(ty => {
                  const val = matrix[tx]?.[ty] ?? 0;
                  const bgColor = getColor(val);
                  const textColor = Math.abs(val) > 0.4 ? '#ffffff' : 'text.primary';
                  return (
                    <TableCell
                      key={ty}
                      align="center"
                      sx={{
                        bgcolor: bgColor,
                        borderRadius: '4px',
                        color: textColor,
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        height: '42px',
                        minWidth: '42px',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        transition: 'all 0.15s ease',
                        cursor: 'default',
                        '&:hover': {
                          transform: 'scale(1.08)',
                          boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
                          zIndex: 10
                        }
                      }}
                      title={`${tx} vs ${ty}: ${val.toFixed(4)}`}
                    >
                      {val.toFixed(2)}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// NetworkGraphChart
// Renders SVG network nodes representing contagion pathways or bipartite relationships
// ---------------------------------------------------------------------------
export function NetworkGraphChart({ nodes = [], edges = [], label = "Ownership Centrality" }) {
  if (!nodes || nodes.length === 0) {
    return <Typography variant="body2" color="text.secondary">No network components available</Typography>;
  }

  // Width and height of SVG viewport
  const width = 500;
  const height = 280;

  // Lay out nodes deterministically: Institutions on left, Assets on right (Bipartite)
  // If asset-only co-ownership graph, lay out in a circle.
  const isBipartite = nodes.some(n => n.nodeType === "Institution");
  const nodePositions = {};

  if (isBipartite) {
    const institutions = nodes.filter(n => n.nodeType === "Institution");
    const assets = nodes.filter(n => n.nodeType === "Asset");

    institutions.forEach((n, idx) => {
      nodePositions[n.id] = {
        x: 120,
        y: height * ((idx + 1) / (institutions.length + 1))
      };
    });

    assets.forEach((n, idx) => {
      nodePositions[n.id] = {
        x: 380,
        y: height * ((idx + 1) / (assets.length + 1))
      };
    });
  } else {
    // Circle layout
    const radius = 90;
    const cx = width / 2;
    const cy = height / 2;
    nodes.forEach((n, idx) => {
      const angle = (idx / nodes.length) * 2 * Math.PI - Math.PI / 2;
      nodePositions[n.id] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle)
      };
    });
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* SVG Graphic */}
      <Box sx={{ bgcolor: '#0D0D0D', borderRadius: 2, p: 1, border: '1px solid #262626', display: 'flex', justifyContent: 'center' }}>
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: '500px' }}>
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563" />
            </marker>
          </defs>

          {/* Render Connections */}
          {edges.map((e, idx) => {
            const start = nodePositions[e.source];
            const end = nodePositions[e.target];
            if (!start || !end) return null;
            const weight = e.edgeWeight || e.coOwnershipWeight || 1.0;
            const strokeWidth = Math.max(1, Math.min(weight * 0.8, 6));
            return (
              <line
                key={`edge-${idx}`}
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke="#374151"
                strokeWidth={strokeWidth}
                opacity={0.65}
                title={`Overlapping concentration: ${weight}`}
              />
            );
          })}

          {/* Render Nodes */}
          {nodes.map(n => {
            const pos = nodePositions[n.id];
            if (!pos) return null;
            const size = n.nodeType === "Institution" ? 14 : 11;
            return (
              <g key={n.id} style={{ cursor: 'pointer' }}>
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={size}
                  fill={n.color || "#4b5563"}
                  stroke="#1f2937"
                  strokeWidth={2}
                />
                <text
                  x={pos.x}
                  y={pos.y - size - 4}
                  textAnchor="middle"
                  fill="#e5e7eb"
                  fontSize="10px"
                  fontWeight="600"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  {n.label || n.id}
                </text>
              </g>
            );
          })}
        </svg>
      </Box>

      {/* Structured Table Fallback */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="caption" sx={{ display: 'block', mb: 1, color: 'text.secondary', fontWeight: 600 }}>
          {label} Details (Table Fallback)
        </Typography>
        <TableContainer component={Paper} variant="outlined" sx={{ bgcolor: '#141414', borderColor: '#262626', maxHeight: '150px' }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ bgcolor: '#1c1c1c', color: 'text.secondary', fontWeight: 600, fontSize: '0.72rem' }}>Source</TableCell>
                <TableCell sx={{ bgcolor: '#1c1c1c', color: 'text.secondary', fontWeight: 600, fontSize: '0.72rem' }}>Target</TableCell>
                <TableCell align="right" sx={{ bgcolor: '#1c1c1c', color: 'text.secondary', fontWeight: 600, fontSize: '0.72rem' }}>Strength / Overlap</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {edges.map((e, idx) => (
                <TableRow key={idx} hover sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                  <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem', py: 0.5 }}>{e.source}</TableCell>
                  <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem', py: 0.5 }}>{e.target}</TableCell>
                  <TableCell align="right" sx={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.75rem', py: 0.5 }}>
                    {(e.edgeWeight !== undefined ? e.edgeWeight : e.coOwnershipWeight * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// TimelineChart
// Renders horizontal block diagrams showing state progressions (Regimes, Pipeline)
// ---------------------------------------------------------------------------
export function TimelineChart({ data = [] }) {
  if (!data || data.length === 0) {
    return <Typography variant="body2" color="text.secondary">No timeline events available</Typography>;
  }

  // Get color for regime
  const getRegimeColor = (regime) => {
    if (regime === "Crisis") return "#ef4444"; // Red
    if (regime === "Elevated") return "#f59e0b"; // Yellow
    return "#10b981"; // Green (Calm)
  };

  // Render a horizontal band of blocks
  return (
    <Box sx={{ width: '100%', py: 1 }}>
      <Box
        sx={{
          height: '24px',
          width: '100%',
          display: 'flex',
          borderRadius: '4px',
          overflow: 'hidden',
          border: '1px solid #262626'
        }}
      >
        {data.map((item, idx) => (
          <Box
            key={idx}
            sx={{
              flex: 1,
              height: '100%',
              bgcolor: getRegimeColor(item.regime),
              opacity: 0.85,
              transition: 'opacity 0.15s ease',
              '&:hover': {
                opacity: 1.0,
                transform: 'scaleY(1.15)',
              }
            }}
            title={`${item.date}: ${item.regime} Regime`}
          />
        ))}
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
        <Typography variant="caption" color="text.secondary">{data[0]?.date}</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 8, height: 8, bgcolor: '#10b981', borderRadius: '50%' }} />
            <Typography variant="caption" color="text.secondary">Calm</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 8, height: 8, bgcolor: '#f59e0b', borderRadius: '50%' }} />
            <Typography variant="caption" color="text.secondary">Elevated</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 8, height: 8, bgcolor: '#ef4444', borderRadius: '50%' }} />
            <Typography variant="caption" color="text.secondary">Crisis</Typography>
          </Box>
        </Box>
        <Typography variant="caption" color="text.secondary">{data[data.length - 1]?.date}</Typography>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// BoxplotLikeChart
// Visualizes return distributions with range bars: line from Min to Max, Q1-Q3 box, Median dot
// ---------------------------------------------------------------------------
export function BoxplotLikeChart({ data = [] }) {
  if (!data || data.length === 0) {
    return <Typography variant="body2" color="text.secondary">No distribution metrics available</Typography>;
  }

  // Find min and max bounds across all series to scale the axis
  const globalMin = Math.min(...data.map(item => item.min));
  const globalMax = Math.max(...data.map(item => item.max));
  const range = globalMax - globalMin || 1.0;

  // Convert value to percentage width offset
  const getPercent = (val) => {
    return ((val - globalMin) / range) * 100;
  };

  return (
    <Box sx={{ width: '100%', py: 1, display: 'flex', flexDirection: 'column', gap: 2.5 }}>
      {data.map((item) => {
        const leftMin = getPercent(item.min);
        const leftMax = getPercent(item.max);
        const leftQ1 = getPercent(item.q1);
        const leftQ3 = getPercent(item.q3);
        const leftMed = getPercent(item.median);

        return (
          <Box key={item.ticker || item.version} sx={{ display: 'flex', alignItems: 'center', minWidth: 0 }}>
            {/* Ticker label */}
            <Typography variant="subtitle2" sx={{ width: '60px', fontWeight: 700, fontSize: '0.78rem', color: 'text.secondary' }}>
              {item.ticker || item.version}
            </Typography>

            {/* Boxplot strip */}
            <Box sx={{ flex: 1, height: '36px', position: 'relative', bgcolor: '#121212', borderRadius: '4px', border: '1px solid #1f1f1f', mx: 1 }}>
              {/* Min-Max Whisker Line */}
              <Box
                sx={{
                  position: 'absolute',
                  top: '18px',
                  left: `${leftMin}%`,
                  right: `${100 - leftMax}%`,
                  height: '2px',
                  bgcolor: '#4b5563'
                }}
              />
              
              {/* Min Tick */}
              <Box sx={{ position: 'absolute', top: '13px', left: `${leftMin}%`, height: '12px', width: '2px', bgcolor: '#4b5563' }} />
              {/* Max Tick */}
              <Box sx={{ position: 'absolute', top: '13px', left: `${leftMax}%`, height: '12px', width: '2px', bgcolor: '#4b5563' }} />

              {/* Q1-Q3 Box */}
              <Box
                sx={{
                  position: 'absolute',
                  top: '8px',
                  left: `${leftQ1}%`,
                  width: `${leftQ3 - leftQ1}%`,
                  height: '20px',
                  bgcolor: 'rgba(59, 130, 246, 0.4)',
                  border: '1.5px solid #3b82f6',
                  borderRadius: '2px',
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    bgcolor: 'rgba(59, 130, 246, 0.65)'
                  }
                }}
                title={`Q1: ${item.q1.toFixed(2)}%, Q3: ${item.q3.toFixed(2)}%`}
              />

              {/* Median Line */}
              <Box
                sx={{
                  position: 'absolute',
                  top: '8px',
                  left: `${leftMed}%`,
                  height: '20px',
                  width: '3px',
                  bgcolor: '#ffffff'
                }}
                title={`Median: ${item.median.toFixed(2)}%`}
              />
            </Box>

            {/* Display stats values summary */}
            <Typography variant="caption" sx={{ width: '70px', color: 'text.secondary', fontSize: '0.7rem', textAlign: 'right', display: { xs: 'none', sm: 'block' } }}>
              Med: {item.median.toFixed(1)}%
            </Typography>
          </Box>
        );
      })}

      {/* Axis Scale guide */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #1f1f1f', pt: 0.5, pl: '60px' }}>
        <Typography variant="caption" color="text.secondary">{globalMin.toFixed(1)}%</Typography>
        <Typography variant="caption" color="text.secondary">{( (globalMin + globalMax) / 2 ).toFixed(1)}%</Typography>
        <Typography variant="caption" color="text.secondary">{globalMax.toFixed(1)}%</Typography>
      </Box>
    </Box>
  );
}
