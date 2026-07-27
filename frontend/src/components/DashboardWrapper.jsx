import React from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, CardHeader } from '@mui/material';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

/**
 * Enhanced Dashboard Wrapper Component
 * Provides consistent styling and layout for dashboard content
 */
const DashboardWrapper = ({ title, children, stats = [] }) => {
  return (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'auto',
        p: 3,
        gap: 3,
      }}
    >
      {/* Header Section */}
      {title && (
        <Box sx={{ mb: 2 }}>
          <Typography
            variant="h4"
            sx={{
              fontSize: '1.75rem',
              fontWeight: 700,
              color: '#ECECEC',
              letterSpacing: '-0.02em',
              mb: 1,
            }}
          >
            {title}
          </Typography>
          <Box
            sx={{
              height: '3px',
              width: '60px',
              background: 'linear-gradient(90deg, #FFFFFF 0%, #B4B4B4 100%)',
              borderRadius: '2px',
            }}
          />
        </Box>
      )}

      {/* Stats Row */}
      {stats && stats.length > 0 && (
        <Grid container spacing={2} sx={{ mb: 2 }}>
          {stats.map((stat, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                sx={{
                  backgroundColor: '#1A1A1A',
                  borderColor: '#404040',
                  border: '1px solid #404040',
                  borderRadius: '12px',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: '#666666',
                    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.5)',
                  },
                }}
              >
                <CardContent sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography
                        variant="caption"
                        sx={{
                          color: '#B4B4B4',
                          fontSize: '0.75rem',
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em',
                          fontWeight: 600,
                        }}
                      >
                        {stat.label}
                      </Typography>
                      <Typography
                        variant="h6"
                        sx={{
                          fontSize: '1.5rem',
                          fontWeight: 700,
                          color: '#ECECEC',
                          mt: 0.5,
                        }}
                      >
                        {stat.value}
                      </Typography>
                      {stat.change && (
                        <Typography
                          variant="caption"
                          sx={{
                            color: stat.change > 0 ? '#22C55E' : '#EF4444',
                            fontSize: '0.75rem',
                            mt: 0.5,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.5,
                          }}
                        >
                          {stat.change > 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                          {Math.abs(stat.change)}%
                        </Typography>
                      )}
                    </Box>
                    {stat.icon && (
                      <Box
                        sx={{
                          width: 48,
                          height: 48,
                          backgroundColor: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: '8px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#B4B4B4',
                        }}
                      >
                        {stat.icon}
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Content Section */}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default DashboardWrapper;
