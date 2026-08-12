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

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");

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
    setResult(null);

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
                    onLoad={() =>
                      console.log(
                        "MRI image loaded"
                      )
                    }
                    onError={(e) =>
                      console.error(
                        "MRI image failed:",
                        e.currentTarget.src
                      )
                    }
                  />

                </div>
              )}

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
                    onLoad={() =>
                      console.log(
                        "Grad-CAM image loaded"
                      )
                    }
                    onError={(e) =>
                      console.error(
                        "Grad-CAM image failed:",
                        e.currentTarget.src
                      )
                    }
                  />
                ) : (
                  <p>
                    Grad-CAM could not be
                    generated for this analysis.
                  </p>
                )}

              </div>

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
                    onLoad={() =>
                      console.log(
                        "LIME image loaded"
                      )
                    }
                    onError={(e) =>
                      console.error(
                        "LIME image failed:",
                        e.currentTarget.src
                      )
                    }
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