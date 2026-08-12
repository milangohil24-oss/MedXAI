import { useAuth } from "../context/AuthContext";

export default function Profile() {
  const { user } = useAuth();

  return (
    <div
      style={{
        padding: "32px",
        color: "#ffffff",
        minHeight: "100%",
      }}
    >
      <h1 style={{ fontSize: "32px", marginBottom: "12px" }}>
        Profile
      </h1>

      <p style={{ color: "#94a3b8" }}>
        Researcher account information.
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
        <p>
          <strong>Name:</strong>{" "}
          {user?.name || "Dr. Researcher"}
        </p>

        <p style={{ marginTop: "16px" }}>
          <strong>Email:</strong>{" "}
          {user?.email || "Not available"}
        </p>
      </div>
    </div>
  );
}