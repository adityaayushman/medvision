"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { getMe, login as apiLogin, signup as apiSignup } from "@/lib/dashboard-api";
import type { DashboardUser } from "@/lib/dashboard-types";

const TOKEN_KEY = "mc-dash-token";

interface AuthState {
  user: DashboardUser | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (orgName: string, email: string, password: string, name?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState>({
  user: null,
  token: null,
  loading: true,
  login: async () => {},
  signup: async () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

/** Owns the dashboard JWT + current user. A stored token is re-validated
 *  against GET /api/auth/me on mount rather than trusted blindly, so a
 *  revoked/expired token doesn't leave stale user state around. */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<DashboardUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (!stored) {
      setLoading(false);
      return;
    }
    getMe(stored)
      .then((u) => {
        setToken(stored);
        setUser(u);
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const body = await apiLogin(email, password);
    localStorage.setItem(TOKEN_KEY, body.access_token);
    setToken(body.access_token);
    setUser(body.user);
  }, []);

  const signup = useCallback(async (orgName: string, email: string, password: string, name?: string) => {
    const body = await apiSignup(orgName, email, password, name);
    localStorage.setItem(TOKEN_KEY, body.access_token);
    setToken(body.access_token);
    setUser(body.user);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
