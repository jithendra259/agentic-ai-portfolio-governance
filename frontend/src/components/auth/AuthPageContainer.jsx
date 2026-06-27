import React, { useEffect, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { SignIn, SignUp } from '@clerk/react';
import { dark } from '@clerk/themes';
import { useAuth } from '../../context/AuthContext';

function readAuthMode() {
  return window.location.hash.toLowerCase().includes('sign-up') ? 'sign-up' : 'sign-in';
}

export default function AuthPageContainer() {
  const { loading } = useAuth();
  const [mode, setMode] = useState(readAuthMode);

  useEffect(() => {
    const handleRouteChange = () => setMode(readAuthMode());
    window.addEventListener('hashchange', handleRouteChange);
    return () => window.removeEventListener('hashchange', handleRouteChange);
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', minHeight: '100vh', justifyContent: 'center', alignItems: 'center', bgcolor: '#0D0D0D' }}>
        <CircularProgress sx={{ color: '#FFFFFF' }} />
      </Box>
    );
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: '#0D0D0D',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: { xs: 2, sm: 3 },
        py: { xs: 4, sm: 6 },
      }}
    >
      {mode === 'sign-up' ? (
        <SignUp
          routing="hash"
          signInUrl="#/sign-in"
          fallbackRedirectUrl="/"
          appearance={{ baseTheme: dark }}
        />
      ) : (
        <SignIn
          routing="hash"
          signUpUrl="#/sign-up"
          fallbackRedirectUrl="/"
          appearance={{ baseTheme: dark }}
        />
      )}
    </Box>
  );
}
