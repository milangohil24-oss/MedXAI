export type UserRole = "doctor" | "patient";

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  created_at?: string;
}

export interface Prediction {
  analysis_id: string;
  filename: string;
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<string, number>;
  gradcam_url?: string;
  lime_url?: string;
  image_url?: string;
}

export interface Analysis {
  id: string;
  user_id: string;
  filename: string;
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<string, number> | string;
  gradcam_url?: string;
  lime_url?: string;
  image_url?: string;
  created_at: string;
}

export interface DashboardStats {
  total_analyses: number;
  average_confidence: number;
  latest_prediction?: string;
  recent: Analysis[];
}