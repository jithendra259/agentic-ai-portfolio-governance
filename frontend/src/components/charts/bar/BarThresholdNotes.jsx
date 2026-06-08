import React from 'react';
import { Box } from '@mui/material';

export default function BarThresholdNotes({ thresholds }) {
  if (!thresholds?.length) return null;
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
      {thresholds.map((threshold) => (
        <Box
          key={`${threshold.name}-${threshold.value}`}
          sx={{
            px: 1,
            py: 0.5,
            border: '1px solid #374151',
            borderRadius: '6px',
            color: '#d1d5db',
            fontSize: 12,
          }}
        >
          {threshold.name}: {threshold.value}
        </Box>
      ))}
    </Box>
  );
}
export { BarThresholdNotes };
