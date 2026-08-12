import React from "react";

interface GlassProps {
  children: React.ReactNode;
  className?: string;
}

export function Glass({ children, className = "" }: GlassProps) {
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

export default Glass;