import React from "react";

interface MedXAILogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
  className?: string;
}

export default function MedXAILogo({
  size = "md",
  showText = true,
  className = "",
}: MedXAILogoProps) {
  const iconSizes = {
    sm: 28,
    md: 40,
    lg: 56,
    xl: 72,
  };

  const textSizes = {
    sm: { title: "text-base", sub: "text-[8px]" },
    md: { title: "text-xl", sub: "text-[9px]" },
    lg: { title: "text-2xl", sub: "text-[10px]" },
    xl: { title: "text-3xl", sub: "text-[11px]" },
  };

  const dim = iconSizes[size] || 40;

  return (
    <div className={`medxai-brand-logo inline-flex items-center gap-3 ${className}`}>
      <div
        className="relative flex items-center justify-center flex-shrink-0 rounded-2xl p-1.5"
        style={{
          width: dim,
          height: dim,
          background:
            "linear-gradient(135deg, rgba(0, 217, 255, 0.18), rgba(139, 92, 246, 0.18))",
          border: "1px solid rgba(0, 217, 255, 0.35)",
          boxShadow: "0 0 20px rgba(0, 217, 255, 0.25), inset 0 0 10px rgba(0, 217, 255, 0.1)",
        }}
      >
        <svg
          width="100%"
          height="100%"
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="medxai-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#00D9FF" />
              <stop offset="50%" stopColor="#38BDF8" />
              <stop offset="100%" stopColor="#8B5CF6" />
            </linearGradient>

            <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Stylized Human Brain Silhouette with Circuit Nodes & Pulse Wave */}
          {/* Left Hemisphere Circuit */}
          <path
            d="M30 14 C22 14, 14 20, 14 30 C14 37, 18 43, 24 47 C26 48, 28 50, 30 52"
            stroke="url(#medxai-grad)"
            strokeWidth="3.5"
            strokeLinecap="round"
            fill="none"
            filter="url(#neon-glow)"
          />
          {/* Right Hemisphere Circuit */}
          <path
            d="M34 14 C42 14, 50 20, 50 30 C50 37, 46 43, 40 47 C38 48, 36 50, 34 52"
            stroke="url(#medxai-grad)"
            strokeWidth="3.5"
            strokeLinecap="round"
            fill="none"
            filter="url(#neon-glow)"
          />

          {/* Central Neural Hemisphere Split */}
          <line
            x1="32"
            y1="12"
            x2="32"
            y2="52"
            stroke="#00D9FF"
            strokeWidth="2.5"
            strokeDasharray="4 3"
            strokeLinecap="round"
          />

          {/* Inner Circuit Connections */}
          <path
            d="M20 28 L28 28 L32 22 L36 34 L40 28 L44 28"
            stroke="#00D9FF"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />

          {/* Neural Nodes */}
          <circle cx="20" cy="28" r="2.5" fill="#00D9FF" />
          <circle cx="28" cy="28" r="2" fill="#8B5CF6" />
          <circle cx="32" cy="22" r="2.5" fill="#38BDF8" />
          <circle cx="36" cy="34" r="2.5" fill="#00D9FF" />
          <circle cx="44" cy="28" r="2.5" fill="#8B5CF6" />
          <circle cx="24" cy="40" r="2" fill="#00D9FF" />
          <circle cx="40" cy="40" r="2" fill="#38BDF8" />

          {/* MRI Scan Signal Beam */}
          <line
            x1="10"
            y1="32"
            x2="54"
            y2="32"
            stroke="url(#medxai-grad)"
            strokeWidth="1.5"
            opacity="0.8"
          />
        </svg>
      </div>

      {showText && (
        <div className="flex flex-col leading-none">
          <span className={`brand-title font-extrabold tracking-widest ${textSizes[size].title}`}>
            MED<span className="text-cyan">X</span>AI
          </span>
          <span className={`brand-sub font-bold tracking-[0.25em] text-cyan uppercase mt-1 ${textSizes[size].sub}`}>
            Spatial Intelligence
          </span>
        </div>
      )}
    </div>
  );
}
