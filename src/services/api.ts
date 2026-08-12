import type {
  Analysis,
  DashboardStats,
  Prediction,
  User,
  UserRole,
} from "../types";


// ============================================================
// API BASE URL
// ============================================================

const envApi = import.meta.env.VITE_API_BASE_URL;

export const API =
  envApi && envApi.trim()
    ? envApi.trim().replace(/\/+$/, "")
    : "https://medxai-backend.onrender.com";

// ============================================================
// TOKEN
// ============================================================

function getToken(): string {
  return (
    localStorage.getItem("medxai_token") || ""
  );
}


// ============================================================
// GENERIC REQUEST
// ============================================================

async function request<T>(
  path: string,
  init: RequestInit = {},
  auth = true
): Promise<T> {

  const headers = new Headers(
    init.headers
  );

  if (
    !(init.body instanceof FormData) &&
    !headers.has("Content-Type")
  ) {
    headers.set(
      "Content-Type",
      "application/json"
    );
  }

  const token = getToken();

  if (auth && token) {
    headers.set(
      "Authorization",
      `Bearer ${token}`
    );
  }

  const response = await fetch(
    `${API}${path}`,
    {
      ...init,
      headers,
    }
  );

  const text =
    await response.text();

  let data: any = null;

  try {

    data = text
      ? JSON.parse(text)
      : null;

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


// ============================================================
// API
// ============================================================

export const api = {

  // ==========================================================
  // REGISTER
  // ==========================================================

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

        body: JSON.stringify({
          name: payload.name,
          email: payload.email,
          password: payload.password,
          role: payload.role,
        }),
      },
      false
    );
  },


  // ==========================================================
  // LOGIN
  //
  // Backend uses OAuth2PasswordRequestForm.
  // Therefore username + password must be sent as
  // application/x-www-form-urlencoded.
  // ==========================================================

  login(payload: {
    email: string;
    password: string;
    role: UserRole;
  }) {

    const form =
      new URLSearchParams();

    form.append(
      "username",
      payload.email
    );

    form.append(
      "password",
      payload.password
    );

    return request<any>(
      "/auth/login",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/x-www-form-urlencoded",
        },

        body: form.toString(),
      },
      false
    );
  },


  // ==========================================================
  // BACKWARD COMPATIBILITY
  // ==========================================================

  doctorLogin(payload: {
    email: string;
    password: string;
  }) {

    return this.login({
      email: payload.email,
      password: payload.password,
      role: "doctor",
    });
  },


  patientLogin(payload: {
    email: string;
    password: string;
  }) {

    return this.login({
      email: payload.email,
      password: payload.password,
      role: "patient",
    });
  },


  // ==========================================================
  // CURRENT USER
  // ==========================================================

  me() {

    return request<User>(
      "/auth/me"
    );
  },


  // ==========================================================
  // LOGOUT
  // ==========================================================

  logout() {

    return request<any>(
      "/auth/logout",
      {
        method: "POST",
      }
    );
  },


  // ==========================================================
  // PREDICTION
  // ==========================================================

  predict(file: File) {

    const form =
      new FormData();

    form.append(
      "file",
      file
    );

    return request<Prediction>(
      "/predict",
      {
        method: "POST",
        body: form,
      }
    );
  },


  // ==========================================================
  // ANALYSES / HISTORY
  // ==========================================================

  analyses() {

    return request<Analysis[]>(
      "/analyses"
    );
  },


  getAnalyses() {

    return request<Analysis[]>(
      "/analyses"
    );
  },


  analysis(id: string) {

    return request<Analysis>(
      `/analyses/${id}`
    );
  },


  deleteAnalysis(id: string) {

    return request<any>(
      `/analyses/${id}`,
      {
        method: "DELETE",
      }
    );
  },


  // ==========================================================
  // DASHBOARD
  // ==========================================================

  stats() {

    return request<DashboardStats>(
      "/dashboard/stats"
    );
  },


  // ==========================================================
  // PROFILE
  // ==========================================================

  profile() {

    return request<User>(
      "/profile"
    );
  },


  updateProfile(
    payload: Partial<User>
  ) {

    return request<User>(
      "/profile",
      {
        method: "PUT",

        body: JSON.stringify(
          payload
        ),
      }
    );
  },


  // ==========================================================
  // CHANGE PASSWORD
  // ==========================================================

  changePassword(payload: {
    current_password: string;
    new_password: string;
  }) {

    return request<any>(
      "/auth/change-password",
      {
        method: "POST",

        body: JSON.stringify(
          payload
        ),
      }
    );
  },


  // ==========================================================
  // REPORTS
  // ==========================================================

  reports() {

    return request<any[]>(
      "/reports"
    );
  },


  createReport(id: string) {

    return request<any>(
      `/reports/${id}`,
      {
        method: "POST",
      }
    );
  },


  // ==========================================================
  // LIME
  // ==========================================================

  lime(id: string) {

    return request<any>(
      "/explain/lime",
      {
        method: "POST",

        body: JSON.stringify({
          analysis_id: id,
        }),
      }
    );
  },


  // ==========================================================
  // IMAGE URL
  // ==========================================================

  imageUrl(url?: string) {

    if (!url) {
      return "";
    }

    const cleanUrl =
      url.trim();

    if (!cleanUrl) {
      return "";
    }

    if (
      cleanUrl.startsWith(
        "http://"
      ) ||
      cleanUrl.startsWith(
        "https://"
      )
    ) {
      return cleanUrl;
    }

    return `${API}${
      cleanUrl.startsWith("/")
        ? ""
        : "/"
    }${cleanUrl}`;
  },


  // ==========================================================
  // REPORT DOWNLOAD URL
  // ==========================================================

  reportUrl(id: string) {

    return `${API}/reports/${id}/download`;
  },
};