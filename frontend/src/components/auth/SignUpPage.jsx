import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
  Alert,
  Paper,
  Link
} from '@mui/material';
import { useAuth } from '../../context/AuthContext';

export default function SignUpPage({ onSwitchToLogin }) {
  const { signup } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [plan, setPlan] = useState('Standard Workspace');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (!name.trim() || !email.trim() || !password) {
      setError('Please fill in all required fields.');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);
    try {
      await signup(name, email, password, plan);
    } catch (err) {
      setError(err.message || 'An error occurred during sign up.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        minHeight: '100vh',
        width: '100vw',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: '#0D0D0D',
        color: '#ECECEC',
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      }}
    >
      <Container maxWidth="xs">
        <Paper
          elevation={0}
          sx={{
            p: 4,
            bgcolor: '#1A1A1A',
            border: '1px solid #404040',
            borderRadius: '12px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          }}
        >
          <Box sx={{ mb: 3, textAlign: 'center' }}>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', mb: 1, color: '#FFFFFF' }}>
              Sign up
            </Typography>
            <Typography variant="body2" sx={{ color: '#B4B4B4' }}>
              Create your Portfolio Governance account
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2, bgcolor: 'rgba(211, 47, 47, 0.1)', color: '#FF8A8A', border: '1px solid #D32F2F', '& .MuiAlert-icon': { color: '#FF8A8A' } }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit} noValidate>
            <Stack spacing={2.5}>
              <TextField
                required
                fullWidth
                label="Full Name"
                name="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                variant="outlined"
                slotProps={{
                  inputLabel: { sx: { color: '#B4B4B4', '&.Mui-focused': { color: '#FFFFFF' } } },
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: '#0D0D0D',
                    color: '#ECECEC',
                    '& fieldset': { borderColor: '#404040' },
                    '&:hover fieldset': { borderColor: '#666666' },
                    '&.Mui-focused fieldset': { borderColor: '#FFFFFF' },
                  },
                }}
              />

              <TextField
                required
                fullWidth
                label="Email Address"
                name="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                variant="outlined"
                slotProps={{
                  inputLabel: { sx: { color: '#B4B4B4', '&.Mui-focused': { color: '#FFFFFF' } } },
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: '#0D0D0D',
                    color: '#ECECEC',
                    '& fieldset': { borderColor: '#404040' },
                    '&:hover fieldset': { borderColor: '#666666' },
                    '&.Mui-focused fieldset': { borderColor: '#FFFFFF' },
                  },
                }}
              />

              <TextField
                required
                fullWidth
                label="Password"
                name="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                variant="outlined"
                slotProps={{
                  inputLabel: { sx: { color: '#B4B4B4', '&.Mui-focused': { color: '#FFFFFF' } } },
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: '#0D0D0D',
                    color: '#ECECEC',
                    '& fieldset': { borderColor: '#404040' },
                    '&:hover fieldset': { borderColor: '#666666' },
                    '&.Mui-focused fieldset': { borderColor: '#FFFFFF' },
                  },
                }}
              />

              <FormControl fullWidth>
                <InputLabel id="plan-select-label" sx={{ color: '#B4B4B4', '&.Mui-focused': { color: '#FFFFFF' } }}>
                  Workspace Plan
                </InputLabel>
                <Select
                  labelId="plan-select-label"
                  id="plan-select"
                  value={plan}
                  label="Workspace Plan"
                  onChange={(e) => setPlan(e.target.value)}
                  sx={{
                    bgcolor: '#0D0D0D',
                    color: '#ECECEC',
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#404040' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#666666' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#FFFFFF' },
                    '& .MuiSvgIcon-root': { color: '#ECECEC' },
                  }}
                  MenuProps={{
                    PaperProps: {
                      sx: {
                        bgcolor: '#1A1A1A',
                        border: '1px solid #404040',
                        color: '#ECECEC',
                        '& .MuiMenuItem-root': {
                          '&:hover': { bgcolor: '#333333' },
                          '&.Mui-selected': { bgcolor: '#444444', '&:hover': { bgcolor: '#555555' } },
                        },
                      },
                    },
                  }}
                >
                  <MenuItem value="Standard Workspace">Standard Workspace</MenuItem>
                  <MenuItem value="Advisory workspace">Advisory Workspace</MenuItem>
                </Select>
              </FormControl>

              <Button
                type="submit"
                fullWidth
                variant="contained"
                disabled={loading}
                sx={{
                  py: 1.5,
                  bgcolor: '#FFFFFF',
                  color: '#0D0D0D',
                  fontWeight: 'bold',
                  borderRadius: '6px',
                  '&:hover': { bgcolor: '#ECECEC' },
                  '&.Mui-disabled': { bgcolor: '#666666', color: '#B4B4B4' },
                }}
              >
                {loading ? 'Signing up...' : 'Sign up'}
              </Button>
            </Stack>

            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#B4B4B4' }}>
                Already have an account?{' '}
                <Link
                  component="button"
                  type="button"
                  variant="body2"
                  onClick={onSwitchToLogin}
                  sx={{
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    textDecoration: 'underline',
                    cursor: 'pointer',
                    '&:hover': { color: '#ECECEC' },
                  }}
                >
                  Sign in
                </Link>
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}
