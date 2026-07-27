import React from 'react';
import { Paper, Box, Typography, Skeleton } from '@mui/material';
import { MoreVertical } from 'lucide-react';

/**
 * Chart Container Component
 * Provides consistent styling and layout for chart components
 */
const ChartContainer = ({
  title,
  subtitle,
  children,
  loading = false,
  height = 400,
  actions = null,
  footer = null,
}) => {
  return (
    <Paper
      elevation={1}
      sx={{
        backgroundColor: '#1A1A1A',
        borderColor: '#404040',
        border: '1px solid #404040',
        borderRadius: '12px',
        overflow: 'hidden',
        transition: 'all 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
        height: 'auto',
        '&:hover': {
          borderColor: '#666666',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.5)',
        },
      }}
    >
      {/* Header */}
      {(title || actions) && (
        <Box
          sx={{
            p: 2,
            borderBottom: '1px solid #404040',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box>
            {title && (
              <Typography
                variant="h6"
                sx={{
                  fontSize: '1rem',
                  fontWeight: 600,
                  color: '#ECECEC',
                  m: 0,
                }}
              >
                {title}
              </Typography>
            )}
            {subtitle && (
              <Typography
                variant="caption"
                sx={{
                  color: '#B4B4B4',
                  fontSize: '0.75rem',
                  mt: 0.5,
                  display: 'block',
                }}
              >
                {subtitle}
              </Typography>
            )}
          </Box>
          {actions && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              {actions}
            </Box>
          )}
        </Box>
      )}

      {/* Content */}
      <Box
        sx={{
          p: 2,
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: height,
          overflow: 'auto',
        }}
      >
        {loading ? (
          <Box sx={{ width: '100%' }}>
            <Skeleton
              variant="rectangular"
              height={height}
              sx={{
                backgroundColor: '#2A2A2A',
                borderRadius: '8px',
              }}
            />
          </Box>
        ) : (
          children
        )}
      </Box>

      {/* Footer */}
      {footer && (
        <Box
          sx={{
            p: 2,
            borderTop: '1px solid #404040',
            backgroundColor: '#121212',
            fontSize: '0.75rem',
            color: '#B4B4B4',
            textAlign: 'center',
          }}
        >
          {footer}
        </Box>
      )}
    </Paper>
  );
};

export default ChartContainer;
