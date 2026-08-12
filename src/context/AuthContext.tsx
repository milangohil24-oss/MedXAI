import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

import type { ReactNode } from "react";
import type { User } from "../types";

import { api } from "../services/api";


// ============================================================
// TYPES
// ============================================================

type Role =
  | "doctor"
  | "patient";


interface AuthContextType {

  user: User | null;

  loading: boolean;

  login: (
    email: string,
    password: string,
    role?: Role
  ) => Promise<void>;

  register: (
    name: string,
    email: string,
    password: string,
    role?: Role
  ) => Promise<void>;

  logout: () => Promise<void>;
}


// ============================================================
// CONTEXT
// ============================================================

const AuthContext =
  createContext<
    AuthContextType | undefined
  >(undefined);


// ============================================================
// TOKEN EXTRACTION
// ============================================================

function extractToken(
  data: any
): string | null {

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


// ============================================================
// USER EXTRACTION
// ============================================================

function extractUser(
  response: any,
  email: string,
  role: Role
): User {

  const serverUser =
    response?.user ??
    response?.data?.user;

  if (serverUser) {

    return {
      ...serverUser,
      email:
        serverUser.email ??
        email,
      role:
        serverUser.role ??
        role,
    };
  }

  return {
    id:
      response?.id ??
      response?.data?.id ??
      "",

    name:
      response?.name ??
      response?.data?.name ??
      (
        role === "doctor"
          ? "Dr. Researcher"
          : "Patient"
      ),

    email,

    role,
  };
}


// ============================================================
// PROVIDER
// ============================================================

export function AuthProvider({
  children,
}: {
  children: ReactNode;
}) {

  const [
    user,
    setUser
  ] = useState<User | null>(
    null
  );

  const [
    loading,
    setLoading
  ] = useState(true);


  // ==========================================================
  // RESTORE SESSION
  // ==========================================================

  useEffect(() => {

    const token =
      localStorage.getItem(
        "medxai_token"
      );

    const savedUser =
      localStorage.getItem(
        "medxai_user"
      );

    if (
      token &&
      savedUser
    ) {

      try {

        const parsedUser =
          JSON.parse(
            savedUser
          );

        setUser(
          parsedUser
        );

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


  // ==========================================================
  // LOGIN
  // ==========================================================

  async function login(
    email: string,
    password: string,
    role: Role = "patient"
  ) {

    const response =
      await api.login({
        email,
        password,
        role,
      });


    const token =
      extractToken(
        response
      );


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
      JSON.stringify(
        loggedUser
      )
    );


    setUser(
      loggedUser
    );
  }


  // ==========================================================
  // REGISTER
  // ==========================================================

  async function register(
    name: string,
    email: string,
    password: string,
    role: Role = "patient"
  ) {

    const response =
      await api.register({
        name,
        email,
        password,
        role,
      });


    const token =
      extractToken(
        response
      );


    if (!token) {

      throw new Error(
        "Registration succeeded but the server did not return an authentication token."
      );

    }


    const registeredUser =
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
      JSON.stringify(
        registeredUser
      )
    );


    setUser(
      registeredUser
    );
  }


  // ==========================================================
  // LOGOUT
  // ==========================================================

  async function logout() {

    try {

      await api.logout();

    } catch {

      // Local logout continues even
      // if server logout fails.

    }


    localStorage.removeItem(
      "medxai_token"
    );

    localStorage.removeItem(
      "medxai_user"
    );


    setUser(null);
  }


  // ==========================================================
  // PROVIDER
  // ==========================================================

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


// ============================================================
// HOOK
// ============================================================

export function useAuth() {

  const context =
    useContext(
      AuthContext
    );

  if (!context) {

    throw new Error(
      "useAuth must be used inside AuthProvider"
    );

  }

  return context;
}