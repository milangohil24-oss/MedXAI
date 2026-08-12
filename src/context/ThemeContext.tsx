import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

import type { ReactNode } from "react";

export type Theme = "dark" | "light";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(
  undefined
);

export function ThemeProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = localStorage.getItem("medxai_theme");

    if (saved === "light" || saved === "dark") {
      return saved;
    }

    return "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem("medxai_theme", theme);
  }, [theme]);

  function toggleTheme() {
    setThemeState((current) =>
      current === "dark" ? "light" : "dark"
    );
  }

  function setTheme(newTheme: Theme) {
    setThemeState(newTheme);
  }

  return (
    <ThemeContext.Provider
      value={{
        theme,
        toggleTheme,
        setTheme,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);

  if (!context) {
    throw new Error(
      "useTheme must be used inside ThemeProvider"
    );
  }

  return context;
}