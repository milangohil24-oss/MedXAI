import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

interface Analysis {
  id: string;
  filename: string;
  user_id: string;
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<string, number>;
  image_url?: string;
  gradcam_url?: string | null;
  lime_url?: string | null;
  created_at: string;
}

interface DashboardStats {
  total_analyses: number;
  average_confidence: number;
  latest_prediction?: string | null;
  recent: Analysis[];
}

export default function Dashboard() {
  const [stats, setStats] =
    useState<DashboardStats | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");

      const token =
        localStorage.getItem("medxai_token");

      if (!token) {
        throw new Error(
          "Please log in to view the dashboard."
        );
      }

      const response = await fetch(
        `${API_URL}/dashboard/stats`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      const data =
        await response.json().catch(
          () => null
        );

      if (!response.ok) {

        if (response.status === 401) {
          localStorage.removeItem(
            "medxai_token"
          );

          localStorage.removeItem(
            "medxai_user"
          );
        }

        throw new Error(
          data?.detail ||
            `Dashboard request failed: ${response.status}`
        );
      }

      console.log(
        "DASHBOARD DATA:",
        data
      );

      setStats(data);

    } catch (err: any) {

      console.error(
        "Dashboard error:",
        err
      );

      setError(
        err?.message ||
          "Unable to load dashboard data."
      );

    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  function formatDate(date: string) {
    try {
      return new Date(
        date
      ).toLocaleString();
    } catch {
      return date;
    }
  }

  function getImageUrl(
    url?: string | null
  ) {
    if (!url) return "";

    if (
      url.startsWith("http://") ||
      url.startsWith("https://")
    ) {
      return url;
    }

    if (url.startsWith("/")) {
      return `${API_URL}${url}`;
    }

    return `${API_URL}/${url}`;
  }

  return (
    <div className="page-container">

      <div className="topbar">

        <div>

          <span className="eyebrow">
            MEDXAI INTELLIGENCE
          </span>

          <h2>
            Dashboard
          </h2>

          <p>
            Monitor your MRI analysis activity,
            model confidence and recent predictions.
          </p>

        </div>

        <div className="status-badge">

          <span className="pulse" />

          {error
            ? "SYSTEM ERROR"
            : "SYSTEM READY"}

        </div>

      </div>

      {error && (
        <div className="auth-error">
          {error}
        </div>
      )}

      <div className="dashboard-stats">

        <div className="glass-panel stat-card">

          <span className="eyebrow">
            TOTAL ANALYSES
          </span>

          <h1>
            {loading
              ? "..."
              : stats?.total_analyses ?? 0}
          </h1>

          <p>
            MRI scans analyzed
          </p>

        </div>

        <div className="glass-panel stat-card">

          <span className="eyebrow">
            AVG. CONFIDENCE
          </span>

          <h1>
            {loading
              ? "..."
              : `${Number(
                  stats?.average_confidence ?? 0
                ).toFixed(2)}%`}
          </h1>

          <p>
            Average model confidence
          </p>

        </div>

        <div className="glass-panel stat-card">

          <span className="eyebrow">
            LATEST PREDICTION
          </span>

          <h1>
            {loading
              ? "..."
              : stats?.latest_prediction ||
                "—"}
          </h1>

          <p>
            Most recent MRI classification
          </p>

        </div>

        <div className="glass-panel stat-card">

          <span className="eyebrow">
            SYSTEM STATUS
          </span>

          <h1>
            {error
              ? "Error"
              : "Ready"}
          </h1>

          <p>
            MEDXAI analysis core
          </p>

        </div>

      </div>

      <div className="glass-panel dashboard-hero">

        <span className="eyebrow">
          MEDXAI ANALYSIS CORE
        </span>

        <h1>
          Turn an MRI into an{" "}
          <em>
            explainable signal.
          </em>
        </h1>

        <p>
          Upload a scan to let EfficientNetB0
          classify the image, then inspect
          Grad-CAM heatmaps and LIME explanations
          through the spatial workspace.
        </p>

        <a
          href="/analyze"
          className="primary"
        >
          Start MRI Analysis
        </a>

      </div>

      <div className="glass-panel">

        <div className="panel-head">

          <div>

            <span className="eyebrow">
              ACTIVITY
            </span>

            <h3>
              Recent Activity
            </h3>

          </div>

          <button
            type="button"
            className="secondary"
            onClick={loadDashboard}
            disabled={loading}
          >
            {loading
              ? "Loading..."
              : "Refresh"}
          </button>

        </div>

        {loading && (
          <div className="result-empty">

            <div className="mini-core spin">
              AI
            </div>

            <h3>
              Loading activity...
            </h3>

            <p>
              Fetching the latest MRI analyses.
            </p>

          </div>
        )}

        {!loading &&
          stats &&
          stats.recent &&
          stats.recent.length === 0 && (

            <div className="result-empty">

              <div className="mini-core">
                AI
              </div>

              <h3>
                No analyses yet
              </h3>

              <p>
                Upload an MRI scan to create
                your first analysis.
              </p>

            </div>
          )}

        {!loading &&
          stats &&
          stats.recent &&
          stats.recent.length > 0 && (

            <div className="recent-list">

              {stats.recent.map(
                (analysis) => (

                  <div
                    className="recent-item"
                    key={analysis.id}
                  >

                    <div className="recent-image">

                      {analysis.image_url ? (

                        <img
                          src={getImageUrl(
                            analysis.image_url
                          )}
                          alt="MRI scan"
                        />

                      ) : (

                        <div className="mini-core">
                          MRI
                        </div>

                      )}

                    </div>

                    <div className="recent-info">

                      <span className="eyebrow">
                        MRI ANALYSIS
                      </span>

                      <h3>
                        {analysis.prediction}
                      </h3>

                      <p>
                        {analysis.filename}
                      </p>

                      <small>
                        {formatDate(
                          analysis.created_at
                        )}
                      </small>

                    </div>

                    <div className="recent-confidence">

                      <strong>
                        {Number(
                          analysis.confidence_percentage
                        ).toFixed(2)}
                        %
                      </strong>

                      <span>
                        Confidence
                      </span>

                    </div>

                  </div>
                )
              )}

            </div>
          )}

      </div>

    </div>
  );
}