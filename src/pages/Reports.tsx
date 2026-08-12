import { useEffect, useState } from "react";
import { Glass } from "../components/Glass";
import { api } from "../services/api";

interface Report {
  id: string;
  analysis_id: string;
  filename: string;
  content?: string;
  download_url?: string;
  created_at: string;
}

export default function Reports() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");

      const data = await api.reports();

      setReports(data || []);
    } catch (err: any) {
      console.error("Reports error:", err);
      setError(err?.message || "Unable to load reports.");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport(report: Report) {
    const token = localStorage.getItem("medxai_token");

    if (!token) {
      setError("Please login again.");
      return;
    }

    const url = api.reportUrl(report.id);

    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="eyebrow">CLINICAL REPORTS</div>
          <h1>Reports</h1>
          <p>Generated Alzheimer's MRI analysis reports.</p>
        </div>

        <button
          className="glass-button"
          onClick={loadReports}
        >
          Refresh
        </button>
      </div>

      {loading && (
        <Glass className="empty large">
          Loading reports...
        </Glass>
      )}

      {!loading && error && (
        <Glass className="empty large">
          {error}
        </Glass>
      )}

      {!loading && !error && reports.length === 0 && (
        <Glass className="empty large">
          <h2>No reports yet</h2>
          <p>
            Analyze an MRI scan and generate a report from the
            analysis result.
          </p>
        </Glass>
      )}

      {!loading && !error && reports.length > 0 && (
        <div className="history-list">
          {reports.map((report) => (
            <Glass
              className="history-card"
              key={report.id}
            >
              <div className="history-main">
                <div className="history-info">
                  <div className="eyebrow">
                    MRI ANALYSIS REPORT
                  </div>

                  <h2>{report.filename}</h2>

                  <div className="history-file">
                    Analysis ID: {report.analysis_id}
                  </div>

                  <div className="history-date">
                    {new Date(
                      report.created_at
                    ).toLocaleString()}
                  </div>

                  <button
                    className="glass-button"
                    onClick={() => downloadReport(report)}
                    style={{ marginTop: "16px" }}
                  >
                    Download PDF
                  </button>
                </div>
              </div>
            </Glass>
          ))}
        </div>
      )}
    </div>
  );
}