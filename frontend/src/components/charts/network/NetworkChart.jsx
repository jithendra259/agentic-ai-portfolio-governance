import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme, alpha } from '@mui/material/styles';
import { useXScale } from '@mui/x-charts/hooks';

function getAnimationSx(animation) {
  if (!animation) return {};
  const duration = animation.duration || '1s';
  const delay = animation.delay || '0s';
  const easing = animation.easing || 'ease-out';
  return { '& *': { transitionDuration: duration, transitionDelay: delay, transitionTimingFunction: easing } };
}

export default function NetworkChartRenderer({ spec }) {
  const height = spec.height !== undefined ? spec.height : 400;
  const nodes = spec.nodes || [];
  const edges = spec.edges || [];
  const containerRef = React.useRef(null);
  const [containerWidth, setContainerWidth] = React.useState(600);
  const [hoveredNodeId, setHoveredNodeId] = React.useState(null);
  const [tooltipPos, setTooltipPos] = React.useState(null);
  React.useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => { if (entries && entries[0]) setContainerWidth(entries[0].contentRect.width || 600); });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);
  const margin = { top: 40, right: 60, bottom: 40, left: 60 };
  const plotWidth = containerWidth - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const { minX, maxX, minY, maxY } = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    Object.values(spec.node_positions || {}).forEach(([x, y]) => { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; });
    return { minX: minX === Infinity ? -1 : minX, maxX: maxX === -Infinity ? 1 : maxX, minY: minY === Infinity ? -1 : minY, maxY: maxY === -Infinity ? 1 : maxY };
  }, [spec.node_positions]);
  const nodeCoords = useMemo(() => {
    const coords = {};
    const rx = maxX - minX || 1; const ry = maxY - minY || 1;
    nodes.forEach((node) => {
      const pos = spec.node_positions?.[node.id] || [0, 0];
      coords[node.id] = { x: margin.left + ((pos[0] - minX) / rx) * plotWidth, y: margin.top + ((pos[1] - minY) / ry) * plotHeight };
    });
    return coords;
  }, [nodes, spec.node_positions, minX, maxX, minY, maxY, plotWidth, plotHeight]);
  const connectedNodes = useMemo(() => {
    if (!hoveredNodeId) return new Set();
    const set = new Set([hoveredNodeId]); edges.forEach((edge) => { if (edge.source === hoveredNodeId) set.add(edge.target); if (edge.target === hoveredNodeId) set.add(edge.source); }); return set;
  }, [hoveredNodeId, edges]);
  const handleNodeMouseEnter = (nodeId, e) => {
    setHoveredNodeId(nodeId);
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
    setTooltipPos({ x: Math.min(mouseX + 15, containerWidth - 220), y: Math.max(mouseY - 40, 10) });
  };
  const activeNodeDetails = useMemo(() => {
    if (!hoveredNodeId) return null;
    const node = nodes.find((n) => n.id === hoveredNodeId);
    if (!node) return null;
    const connections = [];
    edges.forEach((edge) => { if (edge.source === hoveredNodeId) connections.push({ name: edge.target, weight: edge.weight }); else if (edge.target === hoveredNodeId) connections.push({ name: edge.source, weight: edge.weight }); });
    connections.sort((a, b) => b.weight - a.weight);
    return { node, connections };
  }, [hoveredNodeId, nodes, edges]);
  if (nodes.length === 0) return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography variant="body2" sx={{ color: '#9ca3af' }}>No network data available.</Typography></Box>;
  const animationSx = getAnimationSx(spec.animation);
  return (
    <Box ref={containerRef} sx={{ width: '100%', height, minHeight: height, position: 'relative', overflow: 'hidden', ...animationSx }}>
      <svg width={containerWidth} height={height} style={{ display: 'block' }}>
        {edges.map((edge, idx) => {
          const from = nodeCoords[edge.source]; const to = nodeCoords[edge.target]; if (!from || !to) return null;
          const isHighlighted = !hoveredNodeId || (edge.source === hoveredNodeId || edge.target === hoveredNodeId);
          return <line key={`edge-${idx}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={isHighlighted ? '#a5f3fc' : '#374151'} strokeWidth={isHighlighted ? 2 : 1} opacity={isHighlighted ? 0.7 : 0.15} />;
        })}
        {nodes.map((node) => {
          const coords = nodeCoords[node.id]; if (!coords) return null;
          const isStock = node.is_stock; const radius = isStock ? (16 + 18 * (node.risk_score || 0.0)) : 10; const isHighlighted = !hoveredNodeId || connectedNodes.has(node.id); const opacity = isHighlighted ? 1 : 0.25; const nodeColor = isStock ? '#22d3ee' : '#f59e0b'; const nodeBorder = '#e5e7eb';
          return (
            <g key={`node-${node.id}`} onMouseEnter={(e) => handleNodeMouseEnter(node.id, e)} onMouseLeave={() => setHoveredNodeId(null)} style={{ cursor: 'pointer', transition: 'all 0.2s ease', opacity }}>
              {isStock && <circle cx={coords.x} cy={coords.y} r={radius + 4} fill={nodeColor} opacity={hoveredNodeId === node.id ? 0.4 : 0.15} />}
              <circle cx={coords.x} cy={coords.y} r={radius} fill={nodeColor} stroke={nodeBorder} strokeWidth={isStock ? 1.5 : 1.0} />
              {isStock ? <text x={coords.x} y={coords.y + 4} textAnchor="middle" fill="#111827" fontSize={10} fontWeight="bold" pointerEvents="none">{node.id}</text> : <text x={coords.x} y={coords.y - 14} textAnchor="middle" fill="#e5e7eb" fontSize={9} pointerEvents="none">{node.id.length > 20 ? `${node.id.substring(0, 18)}...` : node.id}</text>}
            </g>
          );
        })}
      </svg>
      {hoveredNodeId !== null && tooltipPos && activeNodeDetails && (
        <Box sx={{ position: 'absolute', left: tooltipPos.x, top: tooltipPos.y, pointerEvents: 'none', zIndex: 10, bgcolor: 'rgba(17, 24, 39, 0.95)', border: '1px solid #374151', borderRadius: '6px', p: 1.25, minWidth: 200, maxWidth: 260 }}>
          <Typography variant="caption" sx={{ color: '#e5e7eb', display: 'block', fontWeight: 600, mb: 0.5, borderBottom: '1px solid #374151', pb: 0.5 }}>
            {activeNodeDetails.node.is_stock ? `Stock Ticker: ${activeNodeDetails.node.id}` : `Institution: ${activeNodeDetails.node.id}`}
          </Typography>
          <Box sx={{ fontSize: 11, color: '#9ca3af' }}>
            {activeNodeDetails.node.is_stock && <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between' }}><span>Systemic Risk Score:</span><span style={{ color: '#22d3ee', fontWeight: 600 }}>{(activeNodeDetails.node.risk_score * 100).toFixed(2)}%</span></Box>}
            <Typography variant="caption" sx={{ color: '#e5e7eb', display: 'block', fontWeight: 600, mt: 0.5, mb: 0.25 }}>{activeNodeDetails.node.is_stock ? 'Top Institutional Holders:' : 'Investments:'}</Typography>
            {activeNodeDetails.connections.length > 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {activeNodeDetails.connections.slice(0, 5).map((conn, idx) => (<Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}><span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 130 }}>{conn.name}</span><span style={{ color: '#f59e0b', fontWeight: 500 }}>{(conn.weight * 100).toFixed(2)}%</span></Box>))}
              </Box>
            ) : <span style={{ color: '#6b7280' }}>No connection data</span>}
          </Box>
        </Box>
      )}
    </Box>
  );
}
