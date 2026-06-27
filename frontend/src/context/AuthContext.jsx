import React, { createContext, useCallback, useContext, useEffect, useMemo } from 'react';
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
  const { isLoaded: isAuthLoaded, isSignedIn } = useClerkAuth();
  const { isLoaded: isUserLoaded, user } = useUser();
  const { openSignIn, openSignUp, signOut } = useClerk();
  const token = null;
  const loading = !isAuthLoaded || (isSignedIn && !isUserLoaded);

  const session = useMemo(() => {
    if (loading || !isSignedIn) return null;
    return buildSession(user);
  }, [isSignedIn, loading, user]);

  const getAuthToken = useCallback(async () => {
    return null;
  }, []);

  useEffect(() => {
    localStorage.removeItem(LEGACY_TOKEN_KEY);
  }, [isAuthLoaded, isSignedIn, user?.id]);

  const login = useCallback(async () => openSignIn(), [openSignIn]);
  const signup = useCallback(async () => openSignUp(), [openSignUp]);

  const logout = useCallback(async () => {
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
