import {
  BrainCircuit,
  FileText,
  History as HistoryIcon,
  LayoutDashboard,
  LogOut,
  Moon,
  Settings,
  Sun,
  UserCircle,
} from "lucide-react";

import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import MedXAILogo from "./MedXAILogo";

const navItems = [
  {
    to: "/dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    to: "/analyze",
    label: "Analyze MRI",
    icon: BrainCircuit,
  },
  {
    to: "/history",
    label: "History",
    icon: HistoryIcon,
  },
  {
    to: "/reports",
    label: "Reports",
    icon: FileText,
  },
  {
    to: "/profile",
    label: "Profile",
    icon: UserCircle,
  },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate("/login");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand" style={{ paddingBottom: 16 }}>
          <NavLink to="/dashboard">
            <MedXAILogo />
          </NavLink>
        </div>

        <div className="researcher-card">
          <div className="researcher-avatar">
            <UserCircle size={24} />
          </div>

          <div className="researcher-info">
            <span>
              <i className="status-dot" />
              RESEARCHER
            </span>

            <strong>{user?.name || "Dr. Researcher"}</strong>
          </div>
        </div>

        <nav className="navigation">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `navitem ${isActive ? "active" : ""}`
              }
            >
              <Icon size={19} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `navitem ${isActive ? "active" : ""}`
            }
          >
            <Settings size={19} />
            <span>Settings</span>
          </NavLink>

          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === "dark" ? (
              <Sun size={19} />
            ) : (
              <Moon size={19} />
            )}

            <span>
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </span>
          </button>

          <button
            className="navitem logout-button"
            onClick={handleLogout}
          >
            <LogOut size={19} />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <motion.div
          className="page-container"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
        >
          <Outlet />
        </motion.div>
      </main>
    </div>
  );
}