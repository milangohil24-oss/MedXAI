import { ChangeEvent, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

interface PredictionResult {
  analysis_id: string;
  filename: string;
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<string, number>;
  gradcam_url?: string | null;
  lime_url?: string | null;
  image_url?: string | null;
}

interface ReportResult {
  id: string;
  analysis_id: string;
  filename: string;
  content: string;
  download_url: string;
  created_at: string;
}

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);

  const [report, setReport] = useState<ReportResult | null>(null);

  const [error, setError] = useState("");
  const [reportError, setReportError] = useState("");

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setReport(null);
    setError("");
    setReportError("");

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(URL.createObjectURL(selectedFile));
  }

  function getToken(): string {
    return localStorage.getItem("medxai_token") || "";
  }

  async function handleAnalyze() {
    if (!file) {
      setError("Please select an MRI image first.");
      return;
    }

    const token = getToken();

    if (!token) {
      setError("Please log in before analyzing an MRI.");
      return;
    }

    setLoading(true);
    setError("");
    setReportError("");
    setResult(null);
    setReport(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json().catch(() => ({}));

      if (response.status === 401) {
        localStorage.removeItem("medxai_token");
        localStorage.removeItem("medxai_user");

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail || "MRI analysis failed."
        );
      }

      console.log("MEDXAI prediction response:", data);

      setResult(data);
    } catch (err: any) {
      console.error("MEDXAI analysis error:", err);

      setError(
        err?.message ||
          "Unable to connect to the MEDXAI backend."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerateReport() {
    if (!result?.analysis_id) {
      setReportError(
        "No analysis is available for report generation."
      );
      return;
    }

    const token = getToken();

    if (!token) {
      setReportError(
        "Please log in before generating a report."
      );
      return;
    }

    setReportLoading(true);
    setReportError("");

    try {
      console.log(
        "Generating report for analysis:",
        result.analysis_id
      );

      const response = await fetch(
        `${API_URL}/reports/${result.analysis_id}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      const data = await response.json().catch(() => ({}));

      console.log("MEDXAI report response:", data);

      if (response.status === 401) {
        localStorage.removeItem("medxai_token");
        localStorage.removeItem("medxai_user");

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Report generation failed."
        );
      }

      setReport(data);
    } catch (err: any) {
      console.error(
        "MEDXAI report generation error:",
        err
      );

      setReportError(
        err?.message ||
          "Unable to generate the report."
      );
    } finally {
      setReportLoading(false);
    }
  }

  function handleDownloadReport() {
    if (!report?.id) return;

    const token = getToken();

    if (!token) {
      setReportError(
        "Please log in before downloading the report."
      );
      return;
    }

    const downloadUrl =
      `${API_URL}/reports/${report.id}/download`;

    fetch(downloadUrl, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then(async (response) => {
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));

          throw new Error(
            data?.detail ||
              "Unable to download the report."
          );
        }

        return response.blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download =
          report.filename || "MEDXAI_Report.pdf";

        document.body.appendChild(a);
        a.click();

        a.remove();
        window.URL.revokeObjectURL(url);
      })
      .catch((err) => {
        console.error(
          "Report download error:",
          err
        );

        setReportError(
          err?.message ||
            "Unable to download the report."
        );
      });
  }

  function getImageUrl(url?: string | null): string {
    if (!url) return "";

    const cleanUrl = url.trim();

    if (!cleanUrl) return "";

    if (
      cleanUrl.startsWith("http://") ||
      cleanUrl.startsWith("https://")
    ) {
      return cleanUrl;
    }

    if (cleanUrl.startsWith("/")) {
      return `${API_URL}${cleanUrl}`;
    }

    return `${API_URL}/${cleanUrl}`;
  }

  return (
    <div className="page-container">

      <div className="topbar">
        <div>
          <span className="eyebrow">
            AI ANALYSIS
          </span>

          <h2>
            Analyze MRI
          </h2>

          <p>
            Upload an MRI scan to begin
            Alzheimer’s disease analysis.
          </p>
        </div>

        <div className="status-badge">
          <span className="pulse" />
          MODEL ONLINE
        </div>
      </div>

      <div className="analysis-layout">

        {/* ================= MRI UPLOAD ================= */}

        <div className="glass-panel upload-panel">

          <div className="panel-head">
            <h3>MRI Scan</h3>
          </div>

          <div className="scan-frame">

            {preview ? (
              <>
                <img
                  src={preview}
                  alt="MRI preview"
                />

                {file && (
                  <div className="fileline">
                    {file.name}
                  </div>
                )}
              </>
            ) : (
              <div className="drop">
                <h3>
                  Upload MRI Scan
                </h3>

                <p>
                  Click here to select an MRI
                  image for analysis
                </p>

                <span>
                  JPG, JPEG, PNG, BMP or WEBP
                </span>
              </div>
            )}

            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
              onChange={handleFileChange}
            />
          </div>

          {error && (
            <div className="auth-error">
              {error}
            </div>
          )}

          {file && (
            <button
              type="button"
              className="primary"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading
                ? "Analyzing MRI..."
                : "Analyze MRI"}
            </button>
          )}

        </div>

        {/* ================= RESULT ================= */}

        <div className="glass-panel">

          <div className="panel-head">
            <h3>
              Analysis Result
            </h3>
          </div>

          {!result && !loading && (
            <div className="result-empty">

              <div className="mini-core">
                AI
              </div>

              <h3>
                Awaiting MRI Analysis
              </h3>

              <p>
                Upload an MRI scan and click
                Analyze MRI to generate your
                prediction.
              </p>

            </div>
          )}

          {loading && (
            <div className="result-empty">

              <div className="mini-core spin">
                AI
              </div>

              <h3>
                Analyzing MRI...
              </h3>

              <p>
                EfficientNetB0 is processing
                your scan.
              </p>

            </div>
          )}

          {result && (
            <div>

              {/* ================= PREDICTION ================= */}

              <div className="prediction">

                <div>
                  <span className="eyebrow">
                    PREDICTION
                  </span>

                  <h1>
                    {result.prediction}
                  </h1>
                </div>

                <div className="confidence">

                  <div>
                    {Number(
                      result.confidence_percentage
                    ).toFixed(2)}
                    %
                  </div>

                  <span>
                    Confidence
                  </span>

                </div>

              </div>

              {/* ================= PROBABILITIES ================= */}

              <div className="probabilities">

                {Object.entries(
                  result.probabilities || {}
                ).map(([label, value]) => {

                  const percentage =
                    Number(value) * 100;

                  const width = Math.min(
                    Math.max(percentage, 0),
                    100
                  );

                  return (
                    <div
                      className="prob"
                      key={label}
                    >

                      <div>
                        <span>
                          {label}
                        </span>

                        <strong>
                          {percentage.toFixed(2)}
                          %
                        </strong>
                      </div>

                      <div className="bar">
                        <i
                          style={{
                            width: `${width}%`,
                          }}
                        />
                      </div>

                    </div>
                  );
                })}

              </div>

              {/* ================= REPORT ================= */}

              <div
                className="glass-panel"
                style={{
                  marginTop: "28px",
                  padding: "24px",
                }}
              >

                <div className="panel-head">
                  <div>
                    <span className="eyebrow">
                      CLINICAL REPORT
                    </span>

                    <h3>
                      MRI Analysis Report
                    </h3>
                  </div>
                </div>

                <p
                  style={{
                    color: "#94a3b8",
                    marginBottom: "18px",
                  }}
                >
                  Generate a PDF report containing
                  the MRI prediction, confidence,
                  probability scores and
                  explainability results.
                </p>

                {reportError && (
                  <div className="auth-error">
                    {reportError}
                  </div>
                )}

                {!report && (
                  <button
                    type="button"
                    className="primary"
                    onClick={handleGenerateReport}
                    disabled={reportLoading}
                    style={{
                      width: "100%",
                      marginTop: "10px",
                    }}
                  >
                    {reportLoading
                      ? "Generating PDF Report..."
                      : "Generate PDF Report"}
                  </button>
                )}

                {report && (
                  <div>

                    <div
                      style={{
                        padding: "16px",
                        marginBottom: "16px",
                        borderRadius: "12px",
                        background:
                          "rgba(34,197,94,0.08)",
                        border:
                          "1px solid rgba(34,197,94,0.25)",
                      }}
                    >
                      <strong>
                        Report generated successfully
                      </strong>

                      <div
                        style={{
                          marginTop: "6px",
                          color: "#94a3b8",
                        }}
                      >
                        {report.filename}
                      </div>

                    </div>

                    <button
                      type="button"
                      className="primary"
                      onClick={handleDownloadReport}
                      style={{
                        width: "100%",
                      }}
                    >
                      Download PDF Report
                    </button>

                  </div>
                )}

              </div>

              {/* ================= ORIGINAL MRI ================= */}

              {result.image_url && (
                <div className="explain-preview">

                  <h3>
                    MRI Scan
                  </h3>

                  <img
                    src={getImageUrl(
                      result.image_url
                    )}
                    alt="Uploaded MRI"
                  />

                </div>
              )}

              {/* ================= GRAD CAM ================= */}

              <div className="explain-preview">

                <h3>
                  Grad-CAM Explanation
                </h3>

                <p>
                  Highlighted regions show
                  areas that influenced the
                  model prediction.
                </p>

                {result.gradcam_url ? (
                  <img
                    src={getImageUrl(
                      result.gradcam_url
                    )}
                    alt="Grad-CAM explanation"
                  />
                ) : (
                  <p>
                    Grad-CAM could not be
                    generated for this analysis.
                  </p>
                )}

              </div>

              {/* ================= LIME ================= */}

              <div className="explain-preview">

                <h3>
                  LIME Explanation
                </h3>

                <p>
                  Local explanation generated
                  for this MRI prediction.
                </p>

                {result.lime_url ? (
                  <img
                    src={getImageUrl(
                      result.lime_url
                    )}
                    alt="LIME explanation"
                  />
                ) : (
                  <p>
                    LIME explanation could
                    not be generated.
                  </p>
                )}

              </div>

            </div>
          )}

        </div>

      </div>

    </div>
  );
}