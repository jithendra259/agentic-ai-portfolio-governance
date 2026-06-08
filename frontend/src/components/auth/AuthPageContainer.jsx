import React, { useState } from 'react';
import { Box, Alert, Link, Typography, CircularProgress } from '@mui/material';
import { SignInPage } from '@toolpad/core/SignInPage';
import { useAuth } from '../../context/AuthContext';
import SignUpPage from './SignUpPage';
import { BACKEND_BASE } from '../../config/api';

function SignUpLink({ onClick }) {
  return (
    <Typography variant="body2" sx={{ mt: 2, textAlign: 'center', color: '#B4B4B4', width: '100%' }}>
      Don't have an account?{' '}
      <Link
        component="button"
        type="button"
        variant="body2"
        onClick={onClick}
        sx={{
          color: '#FFFFFF',
          fontWeight: 'bold',
          textDecoration: 'underline',
          cursor: 'pointer',
          '&:hover': { color: '#ECECEC' },
        }}
      >
        Sign up
      </Link>
    </Typography>
  );
}

export default function AuthPageContainer() {
  const { login, loading } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', minHeight: '100vh', justifyContent: 'center', alignItems: 'center', bgcolor: '#0D0D0D' }}>
        <CircularProgress sx={{ color: '#FFFFFF' }} />
      </Box>
    );
  }

  if (isSignUp) {
    return <SignUpPage onSwitchToLogin={() => setIsSignUp(false)} />;
  }

  const providers = [
    { id: 'github', name: 'GitHub' },
    { id: 'google', name: 'Google' },
    { id: 'credentials', name: 'Credentials' },
  ];

  return (
    <SignInPage
      providers={providers}
      signIn={async (provider, formData) => {
        try {
          if (provider.id === 'credentials') {
            const email = formData.get('email');
            const password = formData.get('password');
            await login(email, password);
            return {};
          }
          if (provider.id === 'github' || provider.id === 'google') {
            window.location.href = `${BACKEND_BASE}/api/auth/oauth/login/${provider.id}`;
            return {};
          }
          return { error: 'Unsupported provider' };
        } catch (err) {
          return { error: err.message || 'Incorrect credentials.' };
        }
      }}
      slots={{
        signUpLink: SignUpLink,
      }}
      slotProps={{
        signUpLink: {
          onClick: () => setIsSignUp(true),
        },
        emailField: {
          autoFocus: true,
        },
        passwordField: {},
        form: {
          noValidate: true,
        },
      }}
      sx={{
        bgcolor: '#0D0D0D',
        '& .MuiPaper-root': {
          bgcolor: '#1A1A1A',
          border: '1px solid #404040',
          borderRadius: '12px',
        },
        '& .MuiInputBase-root': {
          bgcolor: '#0D0D0D',
          color: '#ECECEC',
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: '#404040',
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#666666',
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#FFFFFF',
          },
        },
        '& .MuiButton-root': {
          bgcolor: '#FFFFFF',
          color: '#0D0D0D',
          fontWeight: 'bold',
          border: 'none',
          '&:hover': {
            bgcolor: '#ECECEC',
          },
        },
        '& .MuiTypography-root': {
          color: '#ECECEC',
        },
        '& .MuiInputLabel-root': {
          color: '#B4B4B4',
          '&.Mui-focused': {
            color: '#FFFFFF',
          },
        },
      }}
    />
  );
}
