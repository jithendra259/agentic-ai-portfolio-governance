export function getAnimationSx(animation) {
  if (!animation) return {};
  const duration = animation.duration || '1s';
  const delay = animation.delay || '0s';
  const easing = animation.easing || 'ease-out';
  return { '& *': { transitionDuration: duration, transitionDelay: delay, transitionTimingFunction: easing } };
}

export function getBounds(positions) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  Object.values(positions || {}).forEach(([x, y]) => { 
    if (x < minX) minX = x; 
    if (x > maxX) maxX = x; 
    if (y < minY) minY = y; 
    if (y > maxY) maxY = y; 
  });
  return { 
    minX: minX === Infinity ? -1 : minX, 
    maxX: maxX === -Infinity ? 1 : maxX, 
    minY: minY === Infinity ? -1 : minY, 
    maxY: maxY === -Infinity ? 1 : maxY 
  };
}

export function computeNodeCoords({ nodes, positions, bounds, plotWidth, plotHeight, margin }) {
  const coords = {};
  const { minX, maxX, minY, maxY } = bounds;
  const rx = maxX - minX || 1; 
  const ry = maxY - minY || 1;
  nodes.forEach((node) => {
    const pos = positions?.[node.id] || [0, 0];
    coords[node.id] = { 
      x: margin.left + ((pos[0] - minX) / rx) * plotWidth, 
      y: margin.top + ((pos[1] - minY) / ry) * plotHeight 
    };
  });
  return coords;
}

export function getActiveNodeDetails({ hoveredNodeId, nodes, edges }) {
  if (!hoveredNodeId) return null;
  const node = nodes.find((n) => n.id === hoveredNodeId);
  if (!node) return null;
  const connections = [];
  edges.forEach((edge) => { 
    if (edge.source === hoveredNodeId) connections.push({ name: edge.target, weight: edge.weight }); 
    else if (edge.target === hoveredNodeId) connections.push({ name: edge.source, weight: edge.weight }); 
  });
  connections.sort((a, b) => b.weight - a.weight);
  return { node, connections };
}
