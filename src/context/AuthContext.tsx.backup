import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

import type { ReactNode } from "react";
import type { User } from "../types";
import { api } from "../services/api";

interface AuthContextType {
  user: User | null;
  loading: boolean;

  login: (
    email: string,
    password: string,
    role?: "doctor" | "patient"
  ) => Promise<void>;

  register: (
    name: string,
    email: string,
    password: string
  ) => Promise<void>;

  logout: () => Promise<void>;
}

const AuthContext = createContext<
  AuthContextType | undefined
>(undefined);

function extractToken(data: any): string | null {
  return (
    data?.access_token ??
    data?.token ??
    data?.accessToken ??
    data?.jwt ??
    data?.data?.access_token ??
    data?.data?.token ??
    null
  );
}

function extractUser(
  response: any,
  email: string,
  role: "doctor" | "patient"
): User {
  return (
    response?.user ??
    response?.data?.user ?? {
      id: response?.id ?? "",
      name:
        response?.name ??
        response?.data?.name ??
        (role === "doctor"
          ? "Dr. Researcher"
          : "Patient"),
      email,
      role,
    }
  );
}

export function AuthProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [user, setUser] =
    useState<User | null>(null);

  const [loading, setLoading] =
    useState(true);

  useEffect(() => {
    const token =
      localStorage.getItem("medxai_token");

    const savedUser =
      localStorage.getItem("medxai_user");

    if (token && savedUser) {
      try {
        const parsedUser = JSON.parse(
          savedUser
        );

        setUser(parsedUser);
      } catch {
        localStorage.removeItem(
          "medxai_user"
        );
        localStorage.removeItem(
          "medxai_token"
        );
      }
    }

    setLoading(false);
  }, []);

  async function login(
    email: string,
    password: string,
    role: "doctor" | "patient" = "doctor"
  ) {
    const response =
      role === "doctor"
        ? await api.doctorLogin({
            email,
            password,
          })
        : await api.patientLogin({
            email,
            password,
          });

    const token =
      extractToken(response);

    if (!token) {
      throw new Error(
        "Login succeeded but the server did not return an authentication token."
      );
    }

    const loggedUser =
      extractUser(
        response,
        email,
        role
      );

    localStorage.setItem(
      "medxai_token",
      token
    );

    localStorage.setItem(
      "medxai_user",
      JSON.stringify(loggedUser)
    );

    setUser(loggedUser);
  }

  async function register(
    name: string,
    email: string,
    password: string
  ) {
    const response =
      await api.register({
        name,
        email,
        password,
        role: "doctor",
      });

    const token =
      extractToken(response);

    if (token) {
      localStorage.setItem(
        "medxai_token",
        token
      );
    }

    const registeredUser =
      extractUser(
        response,
        email,
        "doctor"
      );

    localStorage.setItem(
      "medxai_user",
      JSON.stringify(registeredUser)
    );

    setUser(registeredUser);
  }

  async function logout() {
    try {
      await api.logout();
    } catch {
      // Server logout failure does not prevent local logout.
    }

    localStorage.removeItem(
      "medxai_token"
    );

    localStorage.removeItem(
      "medxai_user"
    );

    setUser(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context =
    useContext(AuthContext);

  if (!context) {
    throw new Error(
      "useAuth must be used inside AuthProvider"
    );
  }

  return context;
}