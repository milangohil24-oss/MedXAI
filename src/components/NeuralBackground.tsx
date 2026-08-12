import { motion } from "framer-motion";

export default function NeuralBackground() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: -1,
        overflow: "hidden",
        pointerEvents: "none",
        background:
          "radial-gradient(circle at 20% 20%, rgba(14,165,233,0.12), transparent 35%), radial-gradient(circle at 80% 70%, rgba(99,102,241,0.10), transparent 35%)",
      }}
    >
      {Array.from({ length: 18 }).map((_, index) => (
        <motion.div
          key={index}
          animate={{
            y: [0, -20, 0],
            opacity: [0.15, 0.35, 0.15],
          }}
          transition={{
            duration: 4 + (index % 4),
            repeat: Infinity,
            delay: index * 0.15,
          }}
          style={{
            position: "absolute",
            width: 3,
            height: 3,
            borderRadius: "50%",
            background: "#38bdf8",
            left: `${(index * 17) % 100}%`,
            top: `${(index * 29) % 100}%`,
            boxShadow: "0 0 12px rgba(56,189,248,0.8)",
          }}
        />
      ))}
    </div>
  );
}