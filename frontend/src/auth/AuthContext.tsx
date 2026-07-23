// Auth state machine.
//
// On mount: call /auth/me to detect an existing session (cookie set on a
// previous visit, still within JWT exp). On login/logout: delegate to the
// api and update local state. Components consume via the useAuth() hook.

import {
  createContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import * as authApi from "../api/auth";
import { ApiError } from "../api/client";
import type { User } from "../types";

export type AuthState =
  | { status: "checking" }
  | { status: "authenticated"; user: User }
  | { status: "unauthenticated" };

export interface AuthContextValue {
  state: AuthState;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: "checking" });

  // Bootstrap on mount: try /auth/me to detect an existing session.
  useEffect(() => {
    let cancelled = false;
    authApi
      .me()
      .then((user) => {
        if (!cancelled) setState({ status: "authenticated", user });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        // 401 (no/expired/invalid cookie) is the normal "no session" path.
        // Network errors or unexpected statuses also fall through to
        // unauthenticated for now — the login flow can re-establish.
        if (!(err instanceof ApiError) || err.status !== 401) {
          console.warn("auth bootstrap: unexpected error", err);
        }
        setState({ status: "unauthenticated" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = async (username: string, password: string) => {
    const user = await authApi.login(username, password);
    setState({ status: "authenticated", user });
  };

  const logout = async () => {
    // Always clear local state, even if the server-side call fails.
    // The user's intent was to log out; not flipping state would leave
    // them stuck on an "authenticated" view they can't escape.
    try {
      await authApi.logout();
    } catch (err) {
      console.warn("logout: server-side call failed; clearing local state anyway", err);
    }
    setState({ status: "unauthenticated" });
  };

  return (
    <AuthContext.Provider value={{ state, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
