// App routing (Step 4b).
//
// URL is the source of truth for the current destination. Sidebar's
// NavLinks own the active state automatically.
//
// Auth gate:
//   - "checking"        → <Bootstrap />
//   - "unauthenticated" → /login is the only real route; everything else
//                         redirects to /login (preserving the intended
//                         destination as state.from for post-login redirect)
//   - "authenticated"   → AuthenticatedLayout with Outlet rendering the
//                         child route. /login while authenticated falls
//                         through the catch-all and bounces to /.
//
// Admin-only routes (e.g. /admin/users) are wrapped in <AdminGate /> for
// belt-and-suspenders alongside the backend's require_admin dependency.

import { type ReactNode } from "react";
import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";

import { AuthProvider } from "./auth/AuthContext";
import { useAuth } from "./auth/useAuth";
import { HostMetricsProvider } from "./components/HostMetricsProvider";
import { Sidebar } from "./components/Sidebar";
import { ToastProvider } from "./components/Toast";
import { TopBar } from "./components/TopBar";
import { AdminUsersPage } from "./pages/AdminUsersPage";
import { HomePage } from "./pages/HomePage";
import { JobsPage } from "./pages/JobsPage";
import { LoginPage } from "./pages/LoginPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { ModelDetailPage } from "./pages/ModelDetailPage";
import { ModelsPage } from "./pages/ModelsPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { ProjectWorkspacePage } from "./pages/ProjectWorkspacePage";
import { SceneDetailPage } from "./pages/SceneDetailPage";
import { ScenesPage } from "./pages/ScenesPage";

function Bootstrap() {
  return (
    <div className="bootstrap">
      <span className="bootstrap__line">checking session…</span>
    </div>
  );
}

function RedirectToLogin() {
  // Capture the URL the user originally tried to reach, so LoginPage can
  // navigate back to it after a successful sign-in.
  const location = useLocation();
  return <Navigate to="/login" replace state={{ from: location }} />;
}

function AdminGate({ children }: { children: ReactNode }) {
  const { state } = useAuth();
  if (state.status !== "authenticated" || !state.user.is_admin) {
    return <Navigate to="/" replace />;
  }
  return <>{children}</>;
}

function AuthenticatedLayout() {
  const { state } = useAuth();
  if (state.status !== "authenticated") return null;
  const { user } = state;

  return (
    // Host-metrics collection lives at this level so polling starts on
    // login and survives navigating between pages. MonitoringPage reads
    // the buffered series via useHostMetrics() instead of polling on
    // its own.
    <HostMetricsProvider>
      <div className="app-shell">
        <Sidebar isAdmin={user.is_admin} />
        <div className="content">
          <TopBar />
          <main className="content__body">
            <Outlet />
          </main>
        </div>
      </div>
    </HostMetricsProvider>
  );
}

function Router() {
  const { state } = useAuth();

  if (state.status === "checking") {
    return <Bootstrap />;
  }

  if (state.status === "unauthenticated") {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<RedirectToLogin />} />
      </Routes>
    );
  }

  // authenticated
  return (
    <Routes>
      <Route element={<AuthenticatedLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/scenes" element={<ScenesPage />} />
        <Route path="/scenes/:sceneId" element={<SceneDetailPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route
          path="/projects/:projectId"
          element={<ProjectWorkspacePage />}
        />
        <Route path="/models" element={<ModelsPage />} />
        <Route
          path="/models/:architecture"
          element={<ModelDetailPage />}
        />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/monitoring" element={<MonitoringPage />} />
        <Route
          path="/admin/users"
          element={
            <AdminGate>
              <AdminUsersPage />
            </AdminGate>
          }
        />
        {/* Anything else (including /login while authenticated) → home. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        {/* ToastProvider lives inside BrowserRouter so toasts can render
            <Link>s, but outside <Router> so the same toast survives a
            route change. */}
        <ToastProvider>
          <Router />
        </ToastProvider>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
