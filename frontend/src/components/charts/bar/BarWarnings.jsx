import React from 'react';
import { Box, Typography } from '@mui/material';

export default function BarWarnings({ warnings, interpretation }) {
  const notes = [...(warnings || []), interpretation].filter(Boolean);
  if (!notes.length) return null;
  return (
    <Box sx={{ mt: 1, color: '#9ca3af', fontSize: 12, lineHeight: 1.45 }}>
      {notes.map((note) => (
        <Typography key={note} variant="caption" sx={{ display: 'block', color: 'inherit' }}>
          {note}
        </Typography>
      ))}
    </Box>
  );
}
export { BarWarnings };
