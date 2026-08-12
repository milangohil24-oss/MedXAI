import { useEffect, useState } from "react";
import { Glass } from "../components/Glass";
import { api } from "../services/api";
import type { Analysis } from "../types";

export default function History() {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    try {
      setLoading(true);
      setError("");

      const data = await api.getAnalyses();

      setAnalyses(data);
    } catch (err) {
      console.error(err);
      setError("Unable to load analysis history.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="eyebrow">ANALYSIS ARCHIVE</div>
          <h1>History</h1>
          <p>Your previous MRI analyses.</p>
        </div>

        <button className="glass-button" onClick={loadHistory}>
          Refresh
        </button>
      </div>

      {loading && (
        <Glass className="empty large">
          Loading analysis history...
        </Glass>
      )}

      {!loading && error && (
        <Glass className="empty large">
          {error}
        </Glass>
      )}

      {!loading && !error && analyses.length === 0 && (
        <Glass className="empty large">
          No analysis history yet.
        </Glass>
      )}

      {!loading && !error && analyses.length > 0 && (
        <div className="history-list">
          {analyses.map((analysis, index) => (
            <Glass
              className="history-card"
              key={analysis.id ?? `${analysis.filename}-${index}`}
            >
              <div className="history-main">
                <div className="history-image">
                  {analysis.image_url ? (
                    <img
                      src={api.imageUrl(analysis.image_url)}
                      alt="MRI scan"
                    />
                  ) : (
                    <div className="image-placeholder">
                      MRI
                    </div>
                  )}
                </div>

                <div className="history-info">
                  <div className="eyebrow">
                    MRI ANALYSIS
                  </div>

                  <h2>{analysis.prediction}</h2>

                  <div className="history-confidence">
                    {(
                      analysis.confidence_percentage ??
                      analysis.confidence * 100
                    ).toFixed(2)}
                    %
                  </div>

                  <div className="history-file">
                    {analysis.filename}
                  </div>

                  {analysis.created_at && (
                    <div className="history-date">
                      {new Date(
                        analysis.created_at
                      ).toLocaleString()}
                    </div>
                  )}
                </div>
              </div>
            </Glass>
          ))}
        </div>
      )}
    </div>
  );
}