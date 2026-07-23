// Notes + NoteReferences api client (Step 16).

import { ApiError, fetchJson } from "./client";

export type NoteReferenceKind =
  | "project"
  | "action"
  | "output"
  | "viz"
  | "scene";

export interface NoteReference {
  id: string;
  kind: NoteReferenceKind;
  target_id: string;
  created_at: string;
}

export interface Note {
  id: string;
  project_id: string;
  content: string;
  references: NoteReference[];
  created_at: string;
  updated_at: string;
}

export interface NotesPage {
  items: Note[];
  total: number;
  limit: number;
  offset: number;
}

export interface NoteWriteBody {
  content?: string;
  references?: string[];
}

export async function listNotes(projectId: string): Promise<NotesPage> {
  return fetchJson<NotesPage>(
    `/projects/${encodeURIComponent(projectId)}/notes?limit=200`,
  );
}

export async function createNote(
  projectId: string,
  body: NoteWriteBody,
): Promise<Note> {
  return fetchJson<Note>(
    `/projects/${encodeURIComponent(projectId)}/notes`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

export async function patchNote(
  noteId: string,
  body: NoteWriteBody,
): Promise<Note> {
  return fetchJson<Note>(`/notes/${encodeURIComponent(noteId)}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export async function deleteNote(noteId: string): Promise<void> {
  const res = await fetch(`/api/notes/${encodeURIComponent(noteId)}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok && res.status !== 204) throw new ApiError(res.status);
}
