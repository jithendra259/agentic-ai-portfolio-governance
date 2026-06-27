import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useAuth as useClerkAuth, useClerk, useUser } from '@clerk/react';

const AuthContext = createContext(null);
const LEGACY_TOKEN_KEY = 'portfolio-governance-auth-token';

function buildSession(user) {
  if (!user) return null;

  const email = user.primaryEmailAddress?.emailAddress || user.emailAddresses?.[0]?.emailAddress || null;
  const name = user.fullName || user.firstName || email || 'User';
  const plan = user.publicMetadata?.plan || user.unsafeMetadata?.plan || 'Advisory workspace';

  return {
    user: {
      id: user.id,
      name,
      email,
      image: user.imageUrl,
      plan,
    },
  };
}

export function AuthProvider({ children }) {
  const { isLoaded: isAuthLoaded, isSignedIn, getToken: getClerkToken } = useClerkAuth();
  const { isLoaded: isUserLoaded, user } = useUser();
  const { openSignIn, openSignUp, signOut } = useClerk();
  const [token, setToken] = useState(null);
  const loading = !isAuthLoaded || (isSignedIn && !isUserLoaded);

  const session = useMemo(() => {
    if (loading || !isSignedIn) return null;
    return buildSession(user);
  }, [isSignedIn, loading, user]);

  const getAuthToken = useCallback(async () => {
    if (!isAuthLoaded || !isSignedIn) return null;
    try {
      const nextToken = await getClerkToken();
      setToken(nextToken || null);
      return nextToken || null;
    } catch (error) {
      console.error('Unable to read Clerk auth token:', error);
      setToken(null);
      return null;
    }
  }, [getClerkToken, isAuthLoaded, isSignedIn]);

  useEffect(() => {
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    if (!isAuthLoaded || !isSignedIn) {
      setToken(null);
      return undefined;
    }

    let cancelled = false;
    getClerkToken()
      .then(nextToken => {
        if (!cancelled) setToken(nextToken || null);
      })
      .catch(error => {
        if (!cancelled) {
          console.error('Unable to initialize Clerk auth token:', error);
          setToken(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [getClerkToken, isAuthLoaded, isSignedIn, user?.id]);

  const login = useCallback(async () => openSignIn(), [openSignIn]);
  const signup = useCallback(async () => openSignUp(), [openSignUp]);

  const logout = useCallback(async () => {
    setToken(null);
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    await signOut();
  }, [signOut]);

  return (
    <AuthContext.Provider value={{ session, token, getAuthToken, login, logout, signup, loading }}>
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
