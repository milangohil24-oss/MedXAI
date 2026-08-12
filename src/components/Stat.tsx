interface StatProps {
  label: string;
  value: string | number;
  description?: string;
}

export default function Stat({
  label,
  value,
  description,
}: StatProps) {
  return (
    <div
      style={{
        padding: "20px",
        borderRadius: "16px",
        background: "rgba(15, 23, 42, 0.55)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        backdropFilter: "blur(16px)",
      }}
    >
      <div
        style={{
          fontSize: "13px",
          color: "#94a3b8",
          marginBottom: "8px",
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: "28px",
          fontWeight: 700,
          color: "#f8fafc",
        }}
      >
        {value}
      </div>

      {description && (
        <div
          style={{
            marginTop: "6px",
            fontSize: "12px",
            color: "#64748b",
          }}
        >
          {description}
        </div>
      )}
    </div>
  );
}