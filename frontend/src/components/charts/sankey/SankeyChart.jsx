import React, { useMemo, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { SankeyChart } from '@mui/x-charts-premium/SankeyChart';
import { useResponsiveChartWidth } from '../common/useResponsiveChartWidth';
import {
  getResponsiveChartHeight,
  getValueFormatter,
  layoutSankey,
} from './sankeyChartUtils';

export default function SankeyChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 350);
  const [containerRef, containerWidth] = useResponsiveChartWidth(360, 320);
  const [activeId, setActiveId] = useState(null);
  
  const formatValue = useMemo(() => getValueFormatter(spec.valueFormatter || 'none'), [spec.valueFormatter]);

  const layout = useMemo(() => {
    return layoutSankey({ spec, containerWidth, height });
  }, [containerWidth, height, spec]);

  if (!layout.nodes.length || !layout.links.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No Sankey data available.</Typography>
      </Box>
    );
  }

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
