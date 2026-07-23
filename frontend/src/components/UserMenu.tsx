// User menu — circular avatar with initials, click to open a dropdown
// containing the user's identity, theme toggle, and Sign out.
//
// Per storyboard-spec § 4, the top bar's right side has "user icon +
// settings". This consolidates them: the avatar is the user icon; the
// dropdown is the per-user settings surface (currently theme + sign-out;
// will grow as we add per-user preferences).

import { useEffect, useRef, useState } from "react";

import { useAuth } from "../auth/useAuth";
import { useTheme } from "../theme";

function getInitials(name: string): string {
  const parts = name.split(/[\s_\-.]+/).filter(Boolean);
  if (parts.length >= 2) {
    return (parts[0][0]! + parts[1][0]!).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
}

export function UserMenu() {
  const { state, logout } = useAuth();
  const { theme, toggle: toggleTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click + Escape. Effect declared before any conditional
  // return so React's hook order is stable.
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (state.status !== "authenticated") return null;
  const { user } = state;

  const fullName = user.display_name ?? user.username;
  const initials = getInitials(fullName);

  return (
    <div className="usermenu" ref={ref}>
      <button
        type="button"
        className="usermenu__avatar"
        onClick={() => setOpen((o) => !o)}
        aria-label={`Open menu for ${fullName}`}
        aria-expanded={open}
        aria-haspopup="menu"
        title={fullName}
      >
        <span className="usermenu__initials">{initials}</span>
      </button>

      {open && (
        <div className="usermenu__dropdown" role="menu">
          <div className="usermenu__identity">
            <div className="usermenu__name">
              {fullName}
              {user.is_admin && <span className="brand__badge">ADMIN</span>}
            </div>
            <div className="usermenu__email">{user.email}</div>
          </div>

          <div className="usermenu__divider" />

          <button
            type="button"
            role="menuitem"
            className="usermenu__item"
            onClick={() => toggleTheme()}
          >
            <span>Theme</span>
            <span className="usermenu__hint">
              {theme === "light" ? "Light ☼" : "Dark ☾"}
            </span>
          </button>

          <div className="usermenu__divider" />

          <button
            type="button"
            role="menuitem"
            className="usermenu__item usermenu__item--danger"
            onClick={() => {
              setOpen(false);
              void logout();
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
