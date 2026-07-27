import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { Inbox, AlertCircle, HelpCircle } from 'lucide-react';

/**
 * Empty State Component
 * Displays when there's no data or content to show
 */
const EmptyState = ({
  icon = <Inbox size={48} />,
  title = 'No Data',
  description = 'There\'s nothing to display here.',
  action = null,
  actionLabel = 'Get Started',
  type = 'default', // 'default', 'error', 'info'
}) => {
  const getIconColor = () => {
    switch (type) {
      case 'error':
        return '#EF4444';
      case 'info':
        return '#3B82F6';
      default:
        return '#B4B4B4';
    }
  };

  const getBackgroundColor = () => {
    switch (type) {
      case 'error':
        return 'rgba(239, 68, 68, 0.1)';
      case 'info':
        return 'rgba(59, 130, 246, 0.1)';
      default:
        return 'rgba(255, 255, 255, 0.05)';
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        minHeight: 300,
        backgroundColor: getBackgroundColor(),
        borderRadius: '12px',
        border: '1px dashed #404040',
        textAlign: 'center',
      }}
    >
      {/* Icon */}
      <Box
        sx={{
          mb: 2,
          color: getIconColor(),
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {icon}
      </Box>

      {/* Title */}
      <Typography
        variant="h6"
        sx={{
          fontSize: '1.125rem',
          fontWeight: 600,
          color: '#ECECEC',
          mb: 1,
        }}
      >
        {title}
      </Typography>

      {/* Description */}
      <Typography
        variant="body2"
        sx={{
          color: '#B4B4B4',
          fontSize: '0.875rem',
          mb: 3,
          maxWidth: 400,
        }}
      >
        {description}
      </Typography>

      {/* Action Button */}
      {action && (
        <Button
          onClick={action}
          variant="contained"
          sx={{
            backgroundColor: '#FFFFFF',
            color: '#0D0D0D',
            fontWeight: 600,
            fontSize: '0.875rem',
            px: 3,
            py: 1,
            borderRadius: '8px',
            transition: 'all 0.2s ease',
            '&:hover': {
              backgroundColor: '#F5F5F5',
            },
          }}
        >
          {actionLabel}
        </Button>
      )}
    </Box>
  );
};

export default EmptyState;
