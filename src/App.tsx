import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import type { ReactNode } from "react";

import { useAuth } from "./context/AuthContext";

import Layout from "./components/Layout";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Analyze from "./pages/Analyze";
import History from "./pages/History";
import Reports from "./pages/Reports";
import Profile from "./pages/Profile";
import Settings from "./pages/Settings";

function ProtectedRoute({
  children,
  role,
}: {
  children: ReactNode;
  role?: "doctor" | "patient";
}) {
  const { user, loading } = useAuth();

  /*
   * IMPORTANT:
   * Wait until AuthContext checks localStorage.
   * Otherwise React can redirect to /login
   * before the saved user is restored.
   */
  if (loading) {
    return (
      <div className="auth-page">
        <div className="auth-card">
          <div className="auth-logo">
            MEDXAI
          </div>

          <div className="result-empty">
            <div className="mini-core">
              MEDXAI
            </div>

            <p>Loading your session...</p>
          </div>
        </div>
      </div>
    );
  }

  /*
   * No logged-in user
   */
  if (!user) {
    return (
      <Navigate
        to="/login"
        replace
      />
    );
  }

  /*
   * Check role when required
   */
  if (role && user.role !== role) {
    return (
      <Navigate
        to="/dashboard"
        replace
      />
    );
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>

      {/* =========================
          PUBLIC ROUTES
          ========================= */}

      <Route
        path="/login"
        element={<Login />}
      />

      <Route
        path="/register"
        element={<Register />}
      />

      {/* =========================
          PROTECTED APPLICATION
          ========================= */}

      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >

        <Route
          path="/dashboard"
          element={<Dashboard />}
        />

        <Route
          path="/analyze"
          element={<Analyze />}
        />

        <Route
          path="/history"
          element={<History />}
        />

        <Route
          path="/reports"
          element={<Reports />}
        />

        <Route
          path="/profile"
          element={<Profile />}
        />

        <Route
          path="/settings"
          element={<Settings />}
        />

      </Route>

      {/* =========================
          ROOT
          ========================= */}

      <Route
        path="/"
        element={
          <Navigate
            to="/dashboard"
            replace
          />
        }
      />

      {/* =========================
          UNKNOWN ROUTES
          ========================= */}

      <Route
        path="*"
        element={
          <Navigate
            to="/dashboard"
            replace
          />
        }
      />

    </Routes>
  );
}