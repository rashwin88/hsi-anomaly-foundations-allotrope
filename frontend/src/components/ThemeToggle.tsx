// Tiny two-state theme toggle. Sun = currently dark (click to go light).
// Moon = currently light (click to go dark).

import { useTheme } from "../theme";

interface ThemeToggleProps {
  /** If true, position fixed in the top-right corner of the viewport
   *  (used on the LoginPage where there's no top bar to tuck into). */
  floating?: boolean;
}

export function ThemeToggle({ floating = false }: ThemeToggleProps) {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      className={`theme-toggle ${floating ? "theme-toggle--floating" : ""}`}
      onClick={toggle}
      aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
      title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
    >
      {theme === "light" ? "☾" : "☼"}
    </button>
  );
}
