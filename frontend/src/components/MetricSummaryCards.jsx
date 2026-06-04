import React from 'react';
import { Box, Card, CardContent, Typography, Grid, Tooltip } from '@mui/material';

export default function MetricSummaryCards({ metrics = [] }) {
  if (!metrics || metrics.length === 0) return null;

  return (
    <Box sx={{ width: '100%', mb: 3 }}>
      <Grid container spacing={2}>
        {metrics.map((m, idx) => (
          <Grid key={idx} size={{ xs: 12, sm: 6, md: 3 }}>
            <Card
              elevation={1}
              sx={{
                bgcolor: '#141414',
                border: '1px solid #262626',
                borderRadius: '8px',
                transition: 'transform 0.15s ease, border-color 0.15s ease',
                '&:hover': {
                  borderColor: m.color || '#3b82f6',
                  transform: 'translateY(-2px)'
                }
              }}
            >
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600, display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {m.label}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'baseline', mt: 0.5, gap: 1 }}>
                  <Typography variant="h5" sx={{ fontWeight: 800, color: m.color || 'text.primary' }}>
                    {m.value}
                  </Typography>
                  {m.change && (
                    <Typography variant="caption" sx={{ color: m.changeColor || '#10b981', fontWeight: 700 }}>
                      {m.change}
                    </Typography>
                  )}
                </Box>
                {m.helpText && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, fontSize: '0.68rem', lineHeight: 1.25 }}>
                    {m.helpText}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
