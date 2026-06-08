import React from 'react';
import { Box, Typography } from '@mui/material';

export default function NetworkTooltip({ hoveredNodeId, tooltipPos, activeNodeDetails }) {
  if (hoveredNodeId === null || !tooltipPos || !activeNodeDetails) return null;
  return (
    <Box 
      sx={{ 
        position: 'absolute', 
        left: tooltipPos.x, 
        top: tooltipPos.y, 
        pointerEvents: 'none', 
        zIndex: 10, 
        bgcolor: 'rgba(17, 24, 39, 0.95)', 
        border: '1px solid #374151', 
        borderRadius: '6px', 
        p: 1.25, 
        minWidth: 200, 
        maxWidth: 260 
      }}
    >
      <Typography 
        variant="caption" 
        sx={{ 
          color: '#e5e7eb', 
          display: 'block', 
          fontWeight: 600, 
          mb: 0.5, 
          borderBottom: '1px solid #374151', 
          pb: 0.5 
        }}
      >
        {activeNodeDetails.node.is_stock ? `Stock Ticker: ${activeNodeDetails.node.id}` : `Institution: ${activeNodeDetails.node.id}`}
      </Typography>
      <Box sx={{ fontSize: 11, color: '#9ca3af' }}>
        {activeNodeDetails.node.is_stock && (
          <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between' }}>
            <span>Systemic Risk Score:</span>
            <span style={{ color: '#22d3ee', fontWeight: 600 }}>
              {(activeNodeDetails.node.risk_score * 100).toFixed(2)}%
            </span>
          </Box>
        )}
        <Typography 
          variant="caption" 
          sx={{ color: '#e5e7eb', display: 'block', fontWeight: 600, mt: 0.5, mb: 0.25 }}
        >
          {activeNodeDetails.node.is_stock ? 'Top Institutional Holders:' : 'Investments:'}
        </Typography>
        {activeNodeDetails.connections.length > 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {activeNodeDetails.connections.slice(0, 5).map((conn, idx) => (
              <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 130 }}>
                  {conn.name}
                </span>
                <span style={{ color: '#f59e0b', fontWeight: 500 }}>
                  {(conn.weight * 100).toFixed(2)}%
                </span>
              </Box>
            ))}
          </Box>
        ) : (
          <span style={{ color: '#6b7280' }}>No connection data</span>
        )}
      </Box>
    </Box>
  );
}
export { NetworkTooltip };
