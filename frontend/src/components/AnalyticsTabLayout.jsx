import React from 'react';
import { Box, Typography, Chip, Grid, Stack } from '@mui/material';
import { ShieldCheck, AlertTriangle, AlertCircle } from 'lucide-react';

export default function AnalyticsTabLayout({
  title,
  description,
  regime = 'Calm',
  summaryCards,
  children
}) {
  const getRegimeChip = () => {
    if (regime === 'Crisis') {
      return (
        <Chip
          icon={<AlertCircle size={15} color="#ef4444" />}
          label="Regime Status: Crisis (Critical)"
          sx={{
            bgcolor: 'rgba(239, 68, 68, 0.08)',
            color: '#ef4444',
            border: '1px solid rgba(239, 68, 68, 0.25)',
            fontWeight: 700,
            textTransform: 'uppercase',
            fontSize: '0.72rem',
            letterSpacing: '0.3px'
          }}
        />
      );
    }
    if (regime === 'Elevated') {
      return (
        <Chip
          icon={<AlertTriangle size={15} color="#f59e0b" />}
          label="Regime Status: Elevated Risk"
          sx={{
            bgcolor: 'rgba(245, 158, 11, 0.08)',
            color: '#f59e0b',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            fontWeight: 700,
            textTransform: 'uppercase',
            fontSize: '0.72rem',
            letterSpacing: '0.3px'
          }}
        />
      );
    }
    return (
      <Chip
        icon={<ShieldCheck size={15} color="#10b981" />}
        label="Regime Status: Calm (Normal)"
        sx={{
          bgcolor: 'rgba(16, 185, 129, 0.08)',
          color: '#10b981',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          fontWeight: 700,
          textTransform: 'uppercase',
          fontSize: '0.72rem',
          letterSpacing: '0.3px'
        }}
      />
    );
  };

  return (
    <Box sx={{ width: '100%', minWidth: 0, py: 1 }}>
      {/* Header and Regime Info */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'center' }, gap: 2, mb: 3 }}>
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="h5" sx={{ fontWeight: 800, color: 'text.primary', letterSpacing: '-0.02em' }}>
            {title}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, maxWidth: '850px', lineHeight: 1.5 }}>
            {description}
          </Typography>
        </Box>
        <Box sx={{ flexShrink: 0 }}>
          {getRegimeChip()}
        </Box>
      </Box>

      {/* Optional Top Summary Cards */}
      {summaryCards}

      {/* Responsive Grid for Children Plots */}
      <Grid container spacing={3} sx={{ minWidth: 0 }}>
        {React.Children.map(children, (child, idx) => {
          if (!child) return null;
          return (
            <Grid key={idx} item xs={12} md={6} sx={{ minWidth: 0 }}>
              {child}
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}
