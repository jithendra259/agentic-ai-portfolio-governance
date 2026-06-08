import React, { useMemo, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import {
  getAnimationSx,
  getBounds,
  computeNodeCoords,
  getActiveNodeDetails,
} from './networkChartUtils';
import NetworkTooltip from './NetworkTooltip';

export default function NetworkChartRenderer({ spec }) {
  const height = spec.height !== undefined ? spec.height : 400;
  const nodes = spec.nodes || [];
  const edges = spec.edges || [];
  
  const [containerRef, containerWidth] = useResponsiveChartWidth(600, 300);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [tooltipPos, setTooltipPos] = useState(null);

  const margin = { top: 40, right: 60, bottom: 40, left: 60 };
  const plotWidth = containerWidth - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const bounds = useMemo(() => {
    return getBounds(spec.node_positions);
  }, [spec.node_positions]);

  const nodeCoords = useMemo(() => {
    return computeNodeCoords({
      nodes,
      positions: spec.node_positions,
      bounds,
      plotWidth,
      plotHeight,
      margin,
    });
  }, [nodes, spec.node_positions, bounds, plotWidth, plotHeight]);

  const connectedNodes = useMemo(() => {
    if (!hoveredNodeId) return new Set();
    const set = new Set([hoveredNodeId]);
    edges.forEach((edge) => {
      if (edge.source === hoveredNodeId) set.add(edge.target);
      if (edge.target === hoveredNodeId) set.add(edge.source);
    });
    return set;
  }, [hoveredNodeId, edges]);

  const handleNodeMouseEnter = (nodeId, e) => {
    setHoveredNodeId(nodeId);
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    setTooltipPos({
      x: Math.min(mouseX + 15, containerWidth - 220),
      y: Math.max(mouseY - 40, 10),
    });
  };

  const activeNodeDetails = useMemo(() => {
    return getActiveNodeDetails({ hoveredNodeId, nodes, edges });
  }, [hoveredNodeId, nodes, edges]);

  if (nodes.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No network data available.</Typography>
      </Box>
    );
  }

  const animationSx = getAnimationSx(spec.animation);

  return (
    <Box ref={containerRef} sx={{ width: '100%', height, minHeight: height, position: 'relative', overflow: 'hidden', ...animationSx }}>
      <svg width={containerWidth} height={height} style={{ display: 'block' }}>
        {edges.map((edge, idx) => {
          const from = nodeCoords[edge.source];
          const to = nodeCoords[edge.target];
          if (!from || !to) return null;
          const isHighlighted = !hoveredNodeId || (edge.source === hoveredNodeId || edge.target === hoveredNodeId);
          return (
            <line
              key={`edge-${idx}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={isHighlighted ? '#a5f3fc' : '#374151'}
              strokeWidth={isHighlighted ? 2 : 1}
              opacity={isHighlighted ? 0.7 : 0.15}
            />
          );
        })}
        {nodes.map((node) => {
          const coords = nodeCoords[node.id];
          if (!coords) return null;
          const isStock = node.is_stock;
          const radius = isStock ? (16 + 18 * (node.risk_score || 0.0)) : 10;
          const isHighlighted = !hoveredNodeId || connectedNodes.has(node.id);
          const opacity = isHighlighted ? 1 : 0.25;
          const nodeColor = isStock ? '#22d3ee' : '#f59e0b';
          const nodeBorder = '#e5e7eb';
          return (
            <g
              key={`node-${node.id}`}
              onMouseEnter={(e) => handleNodeMouseEnter(node.id, e)}
              onMouseLeave={() => setHoveredNodeId(null)}
              style={{ cursor: 'pointer', transition: 'all 0.2s ease', opacity }}
            >
              {isStock && (
                <circle
                  cx={coords.x}
                  cy={coords.y}
                  r={radius + 4}
                  fill={nodeColor}
                  opacity={hoveredNodeId === node.id ? 0.4 : 0.15}
                />
              )}
              <circle
                cx={coords.x}
                cy={coords.y}
                r={radius}
                fill={nodeColor}
                stroke={nodeBorder}
                strokeWidth={isStock ? 1.5 : 1.0}
              />
              {isStock ? (
                <text x={coords.x} y={coords.y + 4} textAnchor="middle" fill="#111827" fontSize={10} fontWeight="bold" pointerEvents="none">
                  {node.id}
                </text>
              ) : (
                <text x={coords.x} y={coords.y - 14} textAnchor="middle" fill="#e5e7eb" fontSize={9} pointerEvents="none">
                  {node.id.length > 20 ? `${node.id.substring(0, 18)}...` : node.id}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <NetworkTooltip
        hoveredNodeId={hoveredNodeId}
        tooltipPos={tooltipPos}
        activeNodeDetails={activeNodeDetails}
      />
    </Box>
  );
}
