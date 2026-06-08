import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { SankeyChart } from '@mui/x-charts-premium/SankeyChart';

const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getResponsiveChartHeight(spec, fallback = 350) {
  const requested = Number(spec?.height);
  return Number.isFinite(requested) ? Math.max(180, Math.min(requested, 720)) : fallback;
}

function getValueFormatter(format) {
  if (format === 'percent' || format === '%') return (v) => (v == null ? '' : `${v.toFixed(1)}%`);
  if (format === 'decimal') return (v) => (v == null ? '' : Number(v).toFixed(2));
  if (format === 'beta') return (v) => (v == null ? '' : `${Number(v).toFixed(2)} beta`);
  if (format === 'k') return (v) => (v == null ? '' : `${(v / 1000).toFixed(1)}k`);
  if (format === 'currency') return (v) => (v == null ? '' : `$${v.toLocaleString()}`);
  return (v) => {
    if (v == null) return '';
    if (Math.abs(v) >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return Number(v).toFixed(2);
  };
}

export default function SankeyChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 350);
  const containerRef = React.useRef(null);
  const [containerWidth, setContainerWidth] = React.useState(360);
  const [activeId, setActiveId] = React.useState(null);
  const formatValue = useMemo(() => getValueFormatter(spec.valueFormatter || 'none'), [spec.valueFormatter]);
  React.useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) setContainerWidth(Math.max(320, entries[0].contentRect.width || 640));
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);
  const layout = useMemo(() => {
    const nodeWidth = spec.nodeOptions?.width ?? 12;
    const nodePadding = spec.nodeOptions?.padding ?? 18;
    const margin = { top: 34, right: 28, bottom: 28, left: 28 };
    const innerWidth = Math.max(260, containerWidth - margin.left - margin.right);
    const innerHeight = Math.max(180, height - margin.top - margin.bottom);
    const nodesById = new Map();
    (spec.nodes || []).forEach((node, index) => {
      const id = node.id ?? node.label ?? `node-${index}`;
      nodesById.set(id, { id, label: node.label || String(id), color: node.color || PALETTE[index % PALETTE.length] });
    });
    const links = (spec.links || []).map((link, index) => ({ id: `${link.source}-${link.target}-${index}`, source: link.source, target: link.target, value: Math.max(0, toFiniteNumber(link.value)) })).filter((link) => link.source != null && link.target != null && link.value > 0);
    links.forEach((link) => {
      if (!nodesById.has(link.source)) nodesById.set(link.source, { id: link.source, label: String(link.source), color: PALETTE[nodesById.size % PALETTE.length] });
      if (!nodesById.has(link.target)) nodesById.set(link.target, { id: link.target, label: String(link.target), color: PALETTE[nodesById.size % PALETTE.length] });
    });
    const nodes = Array.from(nodesById.values());
    const incoming = new Map(nodes.map((node) => [node.id, 0]));
    const outgoing = new Map(nodes.map((node) => [node.id, 0]));
    links.forEach((link) => { outgoing.set(link.source, (outgoing.get(link.source) || 0) + link.value); incoming.set(link.target, (incoming.get(link.target) || 0) + link.value); });
    const depths = new Map(nodes.map((node) => [node.id, incoming.get(node.id) ? 1 : 0]));
    for (let i = 0; i < nodes.length; i += 1) links.forEach((link) => depths.set(link.target, Math.max(depths.get(link.target) || 0, (depths.get(link.source) || 0) + 1)));
    const maxDepth = Math.max(1, ...depths.values());
    const groups = new Map();
    nodes.forEach((node) => { const depth = depths.get(node.id) || 0; if (!groups.has(depth)) groups.set(depth, []); groups.get(depth).push(node); });
    const valueByNode = new Map(nodes.map((node) => [node.id, Math.max(incoming.get(node.id) || 0, outgoing.get(node.id) || 0, 1)]));
    const maxGroupTotal = Math.max(1, ...Array.from(groups.values()).map((group) => group.reduce((sum, node) => sum + valueByNode.get(node.id), 0)));
    const maxGroupCount = Math.max(1, ...Array.from(groups.values()).map((group) => group.length));
    const valueScale = Math.max(0.0001, (innerHeight - nodePadding * Math.max(0, maxGroupCount - 1)) / maxGroupTotal);
    const positionedNodes = new Map();
    groups.forEach((group, depth) => {
      const groupHeight = group.reduce((sum, node) => sum + valueByNode.get(node.id) * valueScale, 0) + nodePadding * Math.max(0, group.length - 1);
      let y = margin.top + Math.max(0, (innerHeight - groupHeight) / 2);
      group.forEach((node) => {
        const nodeHeight = Math.max(16, valueByNode.get(node.id) * valueScale);
        const x = margin.left + (innerWidth - nodeWidth) * (depth / maxDepth);
        positionedNodes.set(node.id, { ...node, value: valueByNode.get(node.id), x0: x, x1: x + nodeWidth, y0: y, y1: y + nodeHeight, depth });
        y += nodeHeight + nodePadding;
      });
    });
    const sourceOffsets = new Map(nodes.map((node) => [node.id, 0]));
    const targetOffsets = new Map(nodes.map((node) => [node.id, 0]));
    const positionedLinks = links.map((link) => {
      const source = positionedNodes.get(link.source); const target = positionedNodes.get(link.target); const width = Math.max(2, link.value * valueScale);
      const sourceY = source.y0 + (sourceOffsets.get(link.source) || 0) + width / 2;
      const targetY = target.y0 + (targetOffsets.get(link.target) || 0) + width / 2;
      sourceOffsets.set(link.source, (sourceOffsets.get(link.source) || 0) + width);
      targetOffsets.set(link.target, (targetOffsets.get(link.target) || 0) + width);
      const midX = (source.x1 + target.x0) / 2;
      return { ...link, source, target, width, color: target.color, d: `M ${source.x1} ${sourceY} C ${midX} ${sourceY}, ${midX} ${targetY}, ${target.x0} ${targetY}` };
    });
    return { width: containerWidth, height, nodeWidth, nodes: Array.from(positionedNodes.values()), links: positionedLinks };
  }, [containerWidth, height, spec.links, spec.nodeOptions, spec.nodes]);
  if (!layout.nodes.length || !layout.links.length) return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography variant="body2" sx={{ color: '#9ca3af' }}>No Sankey data available.</Typography></Box>;
  return (
    <Box ref={containerRef} sx={{ width: '100%', height, minHeight: height, p: 0.5, position: 'relative' }}>
      <SankeyChart
        height={height}
        series={{
          data: { nodes: layout.nodes, links: layout.links },
          nodeOptions: { showLabels: true, ...(spec.nodeOptions || {}) },
          linkOptions: { opacity: 0.42, color: 'source', ...(spec.linkOptions || {}) },
          valueFormatter: (value) => formatValue(value),
        }}
        margin={spec.margin || { top: 24, right: 24, bottom: 24, left: 24 }}
      />
    </Box>
  );
}
