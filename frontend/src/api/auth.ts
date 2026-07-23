// Allotrope api — auth endpoints.
//
// Wraps POST /auth/login, POST /auth/logout, GET /auth/me into typed calls.
// Each maps 1:1 to a backend route (see backend/allotrope/api/auth.py and
// the corresponding sequence diagrams in final design/diagrams/).

import type { User } from "../types";
import { fetchJson } from "./client";

interface LoginResponse {
  user: User;
}

export async function login(username: string, password: string): Promise<User> {
  const { user } = await fetchJson<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  return user;
}

export async function logout(): Promise<void> {
  await fetchJson<void>("/auth/logout", { method: "POST" });
}

export async function me(): Promise<User> {
  return fetchJson<User>("/auth/me");
}
