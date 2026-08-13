import type {
  Analysis,
  DashboardStats,
  Prediction,
  User,
  UserRole,
} from "../types";

const envApi = import.meta.env.VITE_API_BASE_URL;

export const API =
  envApi && envApi.trim()
    ? envApi.replace(/\/$/, "")
    : "http://127.0.0.1:8000";

function getToken() {
  return localStorage.getItem("medxai_token") || "";
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  auth = true
): Promise<T> {
  const headers = new Headers(init.headers);

  if (!(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const currentToken = getToken();

  if (auth && currentToken) {
    headers.set("Authorization", `Bearer ${currentToken}`);
  }

  const response = await fetch(`${API}${path}`, {
    ...init,
    headers,
  });

  const text = await response.text();

  let data: any = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        data?.error ||
        `Request failed (${response.status})`
    );
  }

  return data as T;
}

export const api = {
  register(payload: {
    name: string;
    email: string;
    password: string;
    role: UserRole;
  }) {
    return request<any>(
      "/auth/register",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      false
    );
  },

  doctorLogin(payload: {
    email: string;
    password: string;
  }) {
    return request<any>(
      "/auth/doctor/login",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      false
    );
  },

  patientLogin(payload: {
    email: string;
    password: string;
  }) {
    return request<any>(
      "/auth/patient/login",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      false
    );
  },

  login(payload: {
    email: string;
    password: string;
    role: UserRole;
  }) {
    if (payload.role === "doctor") {
      return this.doctorLogin({
        email: payload.email,
        password: payload.password,
      });
    }

    return this.patientLogin({
      email: payload.email,
      password: payload.password,
    });
  },

  me() {
    return request<User>("/auth/me");
  },

  logout() {
    return request<any>("/auth/logout", {
      method: "POST",
    });
  },

  predict(file: File) {
    const form = new FormData();
    form.append("file", file);

    return request<Prediction>("/predict", {
      method: "POST",
      body: form,
    });
  },

  analyses() {
    return request<Analysis[]>("/analyses");
  },

  getAnalyses() {
    return request<Analysis[]>("/analyses");
  },

  analysis(id: string) {
    return request<Analysis>(`/analyses/${id}`);
  },

  deleteAnalysis(id: string) {
    return request<any>(`/analyses/${id}`, {
      method: "DELETE",
    });
  },

  stats() {
    return request<DashboardStats>("/dashboard/stats");
  },

  profile() {
    return request<User>("/profile");
  },

  updateProfile(payload: Partial<User>) {
    return request<User>("/profile", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  changePassword(payload: {
    current_password: string;
    new_password: string;
  }) {
    return request<any>("/auth/change-password", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  reports() {
    return request<any[]>("/reports");
  },

  createReport(id: string) {
    return request<any>(`/reports/${id}`, {
      method: "POST",
    });
  },

  lime(id: string) {
    return request<any>("/explain/lime", {
      method: "POST",
      body: JSON.stringify({
        analysis_id: id,
      }),
    });
  },

  imageUrl(url?: string) {
    if (!url) {
      return "";
    }

    const cleanUrl = url.trim();

    if (!cleanUrl) {
      return "";
    }

    if (
      cleanUrl.startsWith("http://") ||
      cleanUrl.startsWith("https://")
    ) {
      return cleanUrl;
    }

    return `${API}${
      cleanUrl.startsWith("/") ? "" : "/"
    }${cleanUrl}`;
  },

  reportUrl(id: string) {
    return `${API}/reports/${id}/download`;
  },
};