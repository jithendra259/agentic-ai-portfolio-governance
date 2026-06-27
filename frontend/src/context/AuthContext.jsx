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
  const [tokenLoaded, setTokenLoaded] = useState(false);
  const loading = !isAuthLoaded || (isSignedIn && (!isUserLoaded || !tokenLoaded));

  const session = useMemo(() => {
    if (loading || !isSignedIn) return null;
    return buildSession(user);
  }, [isSignedIn, loading, user]);

  const getAuthToken = useCallback(async (options = {}) => {
    if (!isAuthLoaded || !isSignedIn) return null;
    try {
      const nextToken = await getClerkToken(options);
      if (!nextToken) {
        const refreshedToken = await getClerkToken({ skipCache: true });
        setToken(refreshedToken || null);
        setTokenLoaded(true);
        return refreshedToken || null;
      }
      setToken(nextToken || null);
      setTokenLoaded(true);
      return nextToken || null;
    } catch (error) {
      console.error('Unable to read Clerk auth token:', error);
      setToken(null);
      setTokenLoaded(true);
      return null;
    }
  }, [getClerkToken, isAuthLoaded, isSignedIn]);

  useEffect(() => {
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    if (!isAuthLoaded || !isSignedIn) {
      setToken(null);
      setTokenLoaded(!isAuthLoaded ? false : true);
      return undefined;
    }

    let cancelled = false;
    setTokenLoaded(false);
    getClerkToken()
      .then(nextToken => {
        if (nextToken || cancelled) return nextToken;
        return getClerkToken({ skipCache: true });
      })
      .then(nextToken => {
        if (!cancelled) {
          setToken(nextToken || null);
          setTokenLoaded(true);
        }
      })
      .catch(error => {
        if (!cancelled) {
          console.error('Unable to initialize Clerk auth token:', error);
          setToken(null);
          setTokenLoaded(true);
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
