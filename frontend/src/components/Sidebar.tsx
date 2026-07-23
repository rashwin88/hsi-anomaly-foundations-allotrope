// Sidebar (storyboard-spec § 4).
//
// Six standard destinations + an admin-only "Users" entry below a divider.
// Active state is owned by NavLink (URL-driven). Keyboard navigation works
// out of the box — anchors are tab-stops; focus-visible styling is in CSS.
//
// Collapse state is persisted in localStorage so the choice survives a
// page reload. When collapsed, only the icons stay visible (label text
// becomes the row's title attribute) and the page content gets back the
// width.

import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

import { Icon, type IconName } from "./Icon";

const COLLAPSE_LS_KEY = "allotrope:sidebar-collapsed";

interface SidebarItem {
  to: string;
  label: string;
  icon: IconName;
  end?: boolean;
}

const STANDARD_ITEMS: SidebarItem[] = [
  { to: "/", label: "Home", icon: "home", end: true },
  { to: "/scenes", label: "Scenes", icon: "scenes" },
  { to: "/projects", label: "Projects", icon: "projects" },
  { to: "/models", label: "Models", icon: "models" },
  { to: "/jobs", label: "Jobs", icon: "jobs" },
  { to: "/monitoring", label: "Monitoring", icon: "monitoring" },
];

interface SidebarProps {
  isAdmin: boolean;
}

function classFor({ isActive }: { isActive: boolean }): string {
  return "sidebar__item" + (isActive ? " sidebar__item--active" : "");
}

export function Sidebar({ isAdmin }: SidebarProps) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem(COLLAPSE_LS_KEY) === "1";
    } catch {
      return false;
    }
  });

  // Persist + reflect onto a body data-attribute so app shell CSS can
  // react to the collapse state without prop-drilling.
  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSE_LS_KEY, collapsed ? "1" : "0");
    } catch {
      /* private mode etc. — ignore */
    }
    document.body.dataset.sidebarCollapsed = collapsed ? "1" : "0";
  }, [collapsed]);

  const toggle = () => setCollapsed((v) => !v);

  return (
    <aside className="sidebar" data-collapsed={collapsed ? "true" : "false"}>
      <header className="sidebar__brand">
        <span className="brand__name">Allotrope</span>
        <span className="brand__version">v0.0.1</span>
        <button
          type="button"
          className="sidebar__collapse-btn"
          onClick={toggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? "›" : "‹"}
        </button>
      </header>

      <nav className="sidebar__nav" aria-label="Primary navigation">
        {STANDARD_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={classFor}
            title={collapsed ? item.label : undefined}
          >
            <Icon name={item.icon} className="sidebar__icon" />
            <span className="sidebar__label">{item.label}</span>
          </NavLink>
        ))}

        {isAdmin && (
          <>
            <div className="sidebar__divider" />
            {!collapsed && (
              <span className="sidebar__section-label">Admin</span>
            )}
            <NavLink
              to="/admin/users"
              className={classFor}
              title={collapsed ? "Users" : undefined}
            >
              <Icon name="users" className="sidebar__icon" />
              <span className="sidebar__label">Users</span>
            </NavLink>
          </>
        )}
      </nav>
    </aside>
  );
}
