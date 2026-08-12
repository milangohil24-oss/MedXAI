import { useTheme } from "../context/ThemeContext";

export default function Settings() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div
      style={{
        padding: "32px",
        color: "#ffffff",
        minHeight: "100%",
      }}
    >
      <h1 style={{ fontSize: "32px", marginBottom: "12px" }}>
        Settings
      </h1>

      <p style={{ color: "#94a3b8" }}>
        Manage your MedXAI application preferences.
      </p>

      <div
        style={{
          marginTop: "30px",
          padding: "28px",
          borderRadius: "16px",
          background: "rgba(15, 23, 42, 0.6)",
          border: "1px solid #1e293b",
          maxWidth: "600px",
        }}
      >
        <h2>Appearance</h2>

        <p style={{ color: "#94a3b8", marginTop: "10px" }}>
          Current theme: {theme}
        </p>

        <button
          onClick={toggleTheme}
          style={{
            marginTop: "20px",
            padding: "12px 22px",
            borderRadius: "10px",
            border: "none",
            background: "#0284c7",
            color: "#ffffff",
            cursor: "pointer",
          }}
        >
          Switch to {theme === "dark" ? "Light" : "Dark"} Mode
        </button>
      </div>
    </div>
  );
}