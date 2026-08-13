
import {
  ChangeEvent,
  useEffect,
  useState,
} from "react";

const API_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "https://medxai-backend.onrender.com"
).replace(/\/+$/, "");

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
  explanation_status?: {
    gradcam?: string;
    lime?: string;
  };
  lime_error?: string | null;
}

interface ReportResult {
  id: string;
  analysis_id: string;
  filename: string;
  content: string;
  download_url: string;
  created_at: string;
}

interface AnalysisHistory {
  id: string;
  analysis_id?: string;
  filename: string;
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<string, number>;
  gradcam_url?: string | null;
  lime_url?: string | null;
  image_url?: string | null;
  created_at: string;
}

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const [result, setResult] =
    useState<PredictionResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [limeLoading, setLimeLoading] = useState(false);
  const [reportLoading, setReportLoading] =
    useState(false);

  const [report, setReport] =
    useState<ReportResult | null>(null);

  const [error, setError] = useState("");
  const [limeError, setLimeError] = useState("");
  const [reportError, setReportError] = useState("");

  const [restoring, setRestoring] = useState(true);

  // ============================================================
  // TOKEN
  // ============================================================

  function getToken(): string {
    return (
      localStorage.getItem("medxai_token") || ""
    );
  }

  // ============================================================
  // IMAGE URL
  // ============================================================

  function getImageUrl(
    url?: string | null
  ): string {
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

  // ============================================================
  // HANDLE AUTH FAILURE
  // ============================================================

  function handleUnauthorized() {
    localStorage.removeItem(
      "medxai_token"
    );

    localStorage.removeItem(
      "medxai_user"
    );
  }

  // ============================================================
  // LOAD EXISTING DASHBOARD / LATEST ANALYSIS
  // ============================================================

  useEffect(() => {
    restoreLatestAnalysis();

    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, []);

  async function restoreLatestAnalysis() {
    const token = getToken();

    if (!token) {
      setRestoring(false);
      return;
    }

    try {
      /*
       * First try localStorage.
       * This makes the result survive a browser refresh
       * immediately.
       */

      const cached =
        localStorage.getItem(
          "medxai_latest_analysis"
        );

      if (cached) {
        try {
          const parsed =
            JSON.parse(cached);

          if (
            parsed &&
            parsed.analysis_id
          ) {
            setResult(parsed);
          }
        } catch {
          localStorage.removeItem(
            "medxai_latest_analysis"
          );
        }
      }

      /*
       * Then fetch the real latest analysis
       * from the backend.
       */

      const response = await fetch(
        `${API_URL}/analyses`,
        {
          method: "GET",
          headers: {
            Authorization:
              `Bearer ${token}`,
          },
        }
      );

      const data =
        await response.json().catch(
          () => []
        );

      if (response.status === 401) {
        handleUnauthorized();

        setResult(null);

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Unable to load previous analyses."
        );
      }

      if (
        Array.isArray(data) &&
        data.length > 0
      ) {
        const latest: AnalysisHistory =
          data[0];

        const restoredResult: PredictionResult =
          {
            analysis_id:
              latest.id ||
              latest.analysis_id ||
              "",

            filename:
              latest.filename ||
              "MRI Scan",

            prediction:
              latest.prediction,

            confidence:
              Number(
                latest.confidence || 0
              ),

            confidence_percentage:
              Number(
                latest.confidence_percentage ||
                  0
              ),

            probabilities:
              latest.probabilities ||
              {},

            gradcam_url:
              latest.gradcam_url ||
              null,

            lime_url:
              latest.lime_url ||
              null,

            image_url:
              latest.image_url ||
              null,
          };

        setResult(
          restoredResult
        );

        localStorage.setItem(
          "medxai_latest_analysis",
          JSON.stringify(
            restoredResult
          )
        );

        /*
         * Restore the uploaded MRI preview
         * from backend if possible.
         */
        if (
          restoredResult.image_url
        ) {
          setPreview(
            getImageUrl(
              restoredResult.image_url
            )
          );
        }

        /*
         * Restore report belonging to
         * this analysis.
         */
        await restoreReport(
          restoredResult.analysis_id,
          token
        );
      }
    } catch (err: any) {
      console.error(
        "MEDXAI restore error:",
        err
      );

      /*
       * Do not destroy cached dashboard
       * just because the backend is temporarily
       * unavailable.
       */
    } finally {
      setRestoring(false);
    }
  }

  // ============================================================
  // RESTORE REPORT
  // ============================================================

  async function restoreReport(
    analysisId: string,
    token: string
  ) {
    try {
      const response = await fetch(
        `${API_URL}/reports`,
        {
          method: "GET",
          headers: {
            Authorization:
              `Bearer ${token}`,
          },
        }
      );

      const data =
        await response
          .json()
          .catch(() => []);

      if (
        response.status === 401
      ) {
        return;
      }

      if (!response.ok) {
        return;
      }

      if (
        Array.isArray(data)
      ) {
        const existingReport =
          data.find(
            (item: ReportResult) =>
              item.analysis_id ===
              analysisId
          );

        if (existingReport) {
          setReport(
            existingReport
          );
        }
      }
    } catch (err) {
      console.error(
        "MEDXAI report restore error:",
        err
      );
    }
  }

  // ============================================================
  // FILE CHANGE
  // ============================================================

  function handleFileChange(
    e: ChangeEvent<HTMLInputElement>
  ) {
    const selectedFile =
      e.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);

    setResult(null);
    setReport(null);

    setError("");
    setLimeError("");
    setReportError("");

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(
      URL.createObjectURL(
        selectedFile
      )
    );

    /*
     * New upload means new analysis.
     */
    localStorage.removeItem(
      "medxai_latest_analysis"
    );
  }

  // ============================================================
  // GENERATE LIME
  // ============================================================

  async function generateLime(
    analysisId: string,
    currentResult: PredictionResult
  ) {
    const token = getToken();

    if (!token) {
      return;
    }

    setLimeLoading(true);
    setLimeError("");

    try {
      console.log(
        "Generating LIME for:",
        analysisId
      );

      const response = await fetch(
        `${API_URL}/explain/lime`,
        {
          method: "POST",

          headers: {
            Authorization:
              `Bearer ${token}`,

            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            analysis_id:
              analysisId,
          }),
        }
      );

      const data =
        await response
          .json()
          .catch(() => ({}));

      console.log(
        "MEDXAI LIME response:",
        data
      );

      if (
        response.status === 401
      ) {
        handleUnauthorized();

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "LIME generation failed."
        );
      }

      /*
       * Backend returns:
       *
       * {
       *   analysis_id,
       *   url,
       *   features
       * }
       */

      const limeUrl =
        data?.url ||
        data?.lime_url ||
        null;

      const updatedResult: PredictionResult =
        {
          ...currentResult,
          lime_url:
            limeUrl,
        };

      setResult(
        updatedResult
      );

      localStorage.setItem(
        "medxai_latest_analysis",
        JSON.stringify(
          updatedResult
        )
      );
    } catch (err: any) {
      console.error(
        "MEDXAI LIME error:",
        err
      );

      setLimeError(
        err?.message ||
          "Unable to generate LIME explanation."
      );
    } finally {
      setLimeLoading(false);
    }
  }

  // ============================================================
  // ANALYZE MRI
  // ============================================================

  async function handleAnalyze() {
    if (!file) {
      setError(
        "Please select an MRI image first."
      );
      return;
    }

    const token = getToken();

    if (!token) {
      setError(
        "Please log in before analyzing an MRI."
      );
      return;
    }

    setLoading(true);

    setError("");
    setLimeError("");
    setReportError("");

    setResult(null);
    setReport(null);

    localStorage.removeItem(
      "medxai_latest_analysis"
    );

    try {
      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      const response =
        await fetch(
          `${API_URL}/predict`,
          {
            method: "POST",

            headers: {
              Authorization:
                `Bearer ${token}`,
            },

            body: formData,
          }
        );

      const data =
        await response
          .json()
          .catch(() => ({}));

      if (
        response.status === 401
      ) {
        handleUnauthorized();

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "MRI analysis failed."
        );
      }

      console.log(
        "MEDXAI prediction response:",
        data
      );

      const predictionResult:
        PredictionResult =
        {
          ...data,

          gradcam_url:
            data.gradcam_url ||
            null,

          lime_url:
            data.lime_url ||
            null,

          image_url:
            data.image_url ||
            null,
        };

      setResult(
        predictionResult
      );

      /*
       * Save latest result so browser refresh
       * does not immediately make the page empty.
       */
      localStorage.setItem(
        "medxai_latest_analysis",
        JSON.stringify(
          predictionResult
        )
      );

      /*
       * LIME is intentionally deferred by
       * the backend /predict endpoint.
       *
       * Therefore call /explain/lime here.
       */
      if (
        predictionResult.analysis_id
      ) {
        generateLime(
          predictionResult.analysis_id,
          predictionResult
        );
      }
    } catch (err: any) {
      console.error(
        "MEDXAI analysis error:",
        err
      );

      setError(
        err?.message ||
          "Unable to connect to the MEDXAI backend."
      );
    } finally {
      setLoading(false);
    }
  }

  // ============================================================
  // GENERATE PDF REPORT
  // ============================================================

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

      const response =
        await fetch(
          `${API_URL}/reports/${result.analysis_id}`,
          {
            method: "POST",

            headers: {
              Authorization:
                `Bearer ${token}`,
            },
          }
        );

      const data =
        await response
          .json()
          .catch(() => ({}));

      console.log(
        "MEDXAI report response:",
        data
      );

      if (
        response.status === 401
      ) {
        handleUnauthorized();

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

  // ============================================================
  // DOWNLOAD REPORT
  // ============================================================

  async function handleDownloadReport() {
    if (!report?.id) {
      return;
    }

    const token = getToken();

    if (!token) {
      setReportError(
        "Please log in before downloading the report."
      );
      return;
    }

    try {
      const downloadUrl =
        `${API_URL}/reports/${report.id}/download`;

      const response =
        await fetch(
          downloadUrl,
          {
            method: "GET",

            headers: {
              Authorization:
                `Bearer ${token}`,
            },
          }
        );

      if (
        response.status === 401
      ) {
        handleUnauthorized();

        throw new Error(
          "Your login session has expired. Please log in again."
        );
      }

      if (!response.ok) {
        const data =
          await response
            .json()
            .catch(() => ({}));

        throw new Error(
          data?.detail ||
            "Unable to download the report."
        );
      }

      const blob =
        await response.blob();

      const url =
        window.URL.createObjectURL(
          blob
        );

      const a =
        document.createElement(
          "a"
        );

      a.href = url;

      a.download =
        report.filename ||
        "MEDXAI_Report.pdf";

      document.body.appendChild(
        a
      );

      a.click();

      a.remove();

      window.URL.revokeObjectURL(
        url
      );
    } catch (err: any) {
      console.error(
        "Report download error:",
        err
      );

      setReportError(
        err?.message ||
          "Unable to download the report."
      );
    }
  }

  // ============================================================
  // PAGE
  // ============================================================

  return (
    <div className="page-container">

      {/* ======================================================
          TOP BAR
      ====================================================== */}

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

      {/* ======================================================
          RESTORING EXISTING ANALYSIS
      ====================================================== */}

      {restoring && (
        <div
          className="glass-panel"
          style={{
            marginBottom: "20px",
            padding: "18px",
            textAlign: "center",
          }}
        >
          <p
            style={{
              margin: 0,
              color: "#94a3b8",
            }}
          >
            Loading your latest MEDXAI
            analysis...
          </p>
        </div>
      )}

      <div className="analysis-layout">

        {/* ====================================================
            MRI UPLOAD
        ==================================================== */}

        <div className="glass-panel upload-panel">

          <div className="panel-head">
            <h3>
              MRI Scan
            </h3>
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
              onChange={
                handleFileChange
              }
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
              onClick={
                handleAnalyze
              }
              disabled={loading}
            >
              {loading
                ? "Analyzing MRI..."
                : "Analyze MRI"}
            </button>
          )}

        </div>

        {/* ====================================================
            RESULT
        ==================================================== */}

        <div className="glass-panel">

          <div className="panel-head">
            <h3>
              Analysis Result
            </h3>
          </div>

          {!result &&
            !loading &&
            !restoring && (
              <div className="result-empty">

                <div className="mini-core">
                  AI
                </div>

                <h3>
                  Awaiting MRI Analysis
                </h3>

                <p>
                  Upload an MRI scan and
                  click Analyze MRI to
                  generate your prediction.
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
                EfficientNetB0 is
                processing your scan.
              </p>

            </div>
          )}

          {result && (
            <div>

              {/* =================================================
                  PREDICTION
              ================================================= */}

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

              {/* =================================================
                  PROBABILITIES
              ================================================= */}

              <div className="probabilities">

                {Object.entries(
                  result.probabilities ||
                    {}
                ).map(
                  ([
                    label,
                    value,
                  ]) => {

                    const percentage =
                      Number(value) *
                      100;

                    const width =
                      Math.min(
                        Math.max(
                          percentage,
                          0
                        ),
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
                            {percentage.toFixed(
                              2
                            )}
                            %
                          </strong>

                        </div>

                        <div className="bar">

                          <i
                            style={{
                              width:
                                `${width}%`,
                            }}
                          />

                        </div>

                      </div>
                    );
                  }
                )}

              </div>

              {/* =================================================
                  ORIGINAL MRI
              ================================================= */}

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

              {/* =================================================
                  GRAD-CAM
              ================================================= */}

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

              {/* =================================================
                  LIME
              ================================================= */}

              <div className="explain-preview">

                <h3>
                  LIME Explanation
                </h3>

                <p>
                  LIME identifies local image
                  regions that contributed to
                  the model prediction.
                </p>

                {limeLoading && (
                  <div
                    style={{
                      padding:
                        "18px",
                      marginTop:
                        "12px",
                      borderRadius:
                        "12px",
                      background:
                        "rgba(59,130,246,0.08)",
                      border:
                        "1px solid rgba(59,130,246,0.2)",
                    }}
                  >
                    <p
                      style={{
                        margin: 0,
                        color:
                          "#94a3b8",
                      }}
                    >
                      Generating LIME
                      explanation...
                      This may take a few
                      seconds.
                    </p>
                  </div>
                )}

                {limeError && (
                  <div className="auth-error">
                    {limeError}
                  </div>
                )}

                {result.lime_url ? (
                  <img
                    src={getImageUrl(
                      result.lime_url
                    )}
                    alt="LIME explanation"
                  />
                ) : (
                  !limeLoading &&
                  !limeError && (
                    <p>
                      LIME explanation is
                      not available yet.
                    </p>
                  )
                )}

              </div>

              {/* =================================================
                  CLINICAL REPORT
              ================================================= */}

              <div
                className="glass-panel"
                style={{
                  marginTop:
                    "28px",
                  padding:
                    "24px",
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
                    color:
                      "#94a3b8",
                    marginBottom:
                      "18px",
                  }}
                >
                  Generate a PDF report
                  containing the MRI
                  prediction, confidence,
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
                    onClick={
                      handleGenerateReport
                    }
                    disabled={
                      reportLoading
                    }
                    style={{
                      width:
                        "100%",
                      marginTop:
                        "10px",
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
                        padding:
                          "16px",
                        marginBottom:
                          "16px",
                        borderRadius:
                          "12px",
                        background:
                          "rgba(34,197,94,0.08)",
                        border:
                          "1px solid rgba(34,197,94,0.25)",
                      }}
                    >

                      <strong>
                        Report generated
                        successfully
                      </strong>

                      <div
                        style={{
                          marginTop:
                            "6px",
                          color:
                            "#94a3b8",
                        }}
                      >
                        {report.filename}
                      </div>

                    </div>

                    <button
                      type="button"
                      className="primary"
                      onClick={
                        handleDownloadReport
                      }
                      style={{
                        width:
                          "100%",
                      }}
                    >
                      Download PDF Report
                    </button>

                  </div>
                )}

              </div>

            </div>
          )}

        </div>

      </div>

    </div>
  );
}

