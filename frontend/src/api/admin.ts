// Allotrope api — admin endpoints.
//
// Currently only POST /admin/users (admin-only create user).
// Backend route: backend/allotrope/api/admin.py
// Sequence diagram: final design/diagrams/admin-create-user.drawio

import type { User } from "../types";
import { fetchJson } from "./client";

export interface CreateUserPayload {
  username: string;
  email: string;
  password: string;
  display_name?: string;
  is_admin?: boolean;
}

export async function createUser(payload: CreateUserPayload): Promise<User> {
  return fetchJson<User>("/admin/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
