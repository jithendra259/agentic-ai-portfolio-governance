import React, { createContext, useContext, useState, useEffect } from 'react';
import { BACKEND_BASE } from '../config/api';

const AuthContext = createContext(null);

/**
 * Read token from URL params first (OAuth callback), then fall back to localStorage.
 * This runs synchronously during state initialization so there's no race condition.
 */
function getInitialToken() {
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const tokenFromUrl = urlParams.get('token');
    if (tokenFromUrl) {
      // Store it immediately and clear the URL
      localStorage.setItem('portfolio-governance-auth-token', tokenFromUrl);
      // Replace history without query string — done once synchronously
      window.history.replaceState({}, document.title, window.location.pathname);
      return tokenFromUrl;
    }
  } catch (_) {
    // SSR or restricted environment — ignore
  }
  return localStorage.getItem('portfolio-governance-auth-token');
}

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  // Token initializer reads URL ?token= first (OAuth callback), then localStorage
  const [token, setToken] = useState(getInitialToken);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      localStorage.setItem('portfolio-governance-auth-token', token);
      // Verify token with backend session endpoint
      fetch(`${BACKEND_BASE}/api/auth/session`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(res => {
          if (!res.ok) throw new Error('Session invalid');
          return res.json();
        })
        .then(data => {
          if (data?.session) {
            setSession(data.session);
          } else {
            setSession(null);
            setToken(null);
            localStorage.removeItem('portfolio-governance-auth-token');
          }
          setLoading(false);
        })
        .catch(() => {
          setSession(null);
          setToken(null);
          localStorage.removeItem('portfolio-governance-auth-token');
          setLoading(false);
        });
    } else {
      localStorage.removeItem('portfolio-governance-auth-token');
      setSession(null);
      setLoading(false);
    }
  }, [token]);

  const login = async (email, password) => {
    const res = await fetch(`${BACKEND_BASE}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email, password })
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Incorrect credentials.');
    }
    
    const data = await res.json();
    setToken(data.token);
    setSession(data.session);
    return data.session;
  };

  const signup = async (name, email, password, plan) => {
    const res = await fetch(`${BACKEND_BASE}/api/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ name, email, password, plan })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Sign up failed.');
    }

    const data = await res.json();
    setToken(data.token);
    setSession(data.session);
    return data.session;
  };

  const logout = async () => {
    try {
      await fetch(`${BACKEND_BASE}/api/auth/logout`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
    } catch (e) {
      console.error('Logout request failed:', e);
    }
    setToken(null);
    setSession(null);
    localStorage.removeItem('portfolio-governance-auth-token');
  };

  return (
    <AuthContext.Provider value={{ session, token, login, logout, signup, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
