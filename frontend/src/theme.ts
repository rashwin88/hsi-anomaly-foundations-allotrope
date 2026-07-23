// Theme: light (default) / dark.
//
// Persistence: localStorage["allotrope-theme"] = "light" | "dark".
// Application: data-theme attribute on <html>.
//
// First-paint flicker is prevented by the inline script in index.html that
// reads localStorage and sets data-theme BEFORE React mounts. This hook
// keeps the React state and the DOM attribute in sync after that.

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "allotrope-theme";

function readInitial(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(readInitial);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  return {
    theme,
    toggle: () => setTheme((t) => (t === "light" ? "dark" : "light")),
  };
}
