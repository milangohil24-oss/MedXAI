import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Activity,
  BrainCircuit,
  Lock,
  Mail,
  ShieldCheck,
  Stethoscope,
  UserRound,
} from "lucide-react";

import { useAuth } from "../context/AuthContext";
import type { UserRole } from "../types";

export default function Register() {
  const navigate = useNavigate();
  const { register } = useAuth();

  const [role, setRole] = useState<UserRole>("patient");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    setError("");
    setLoading(true);

    try {
      await register(name, email, password, role);

      navigate("/dashboard", { replace: true });
    } catch (err: any) {
      setError(err?.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">

      {/* LEFT MEDICAL AI SECTION */}
      <section className="auth-visual">
        <div className="auth-visual-overlay" />

        <div className="auth-visual-content">

          <div className="auth-visual-logo">
            <div className="auth-visual-logo-icon">
              <BrainCircuit size={28} />
            </div>

            <div>
              <strong>MEDXAI</strong>
              <span>MEDICAL AI INTELLIGENCE</span>
            </div>
          </div>

          <div className="medical-orb">
            <div className="orb-ring orb-ring-1" />
            <div className="orb-ring orb-ring-2" />
            <div className="orb-ring orb-ring-3" />

            <div className="orb-core">
              <BrainCircuit size={72} />
            </div>

            <div className="scan-line" />
          </div>

          <div className="auth-visual-text">

            <span className="visual-eyebrow">
              <Activity size={14} />
              AI-POWERED NEUROIMAGING
            </span>

            <h1>
              Intelligent
              <br />
              <span>Alzheimer's</span>
              <br />
              Detection
            </h1>

            <p>
              Analyze MRI scans with advanced AI and understand
              predictions through explainable medical intelligence.
            </p>

          </div>

          <div className="medical-features">

            <div className="medical-feature">
              <div className="medical-feature-icon">
                <BrainCircuit size={18} />
              </div>

              <div>
                <strong>AI Analysis</strong>
                <span>EfficientNet-based detection</span>
              </div>
            </div>

            <div className="medical-feature">
              <div className="medical-feature-icon">
                <ShieldCheck size={18} />
              </div>

              <div>
                <strong>Explainable AI</strong>
                <span>Transparent model predictions</span>
              </div>
            </div>

          </div>

          <div className="visual-footer">
            Intelligent healthcare powered by artificial intelligence
          </div>

        </div>

        <div className="medical-grid" />
        <div className="medical-particle particle-1" />
        <div className="medical-particle particle-2" />
        <div className="medical-particle particle-3" />
        <div className="medical-particle particle-4" />

      </section>


      {/* RIGHT REGISTER SECTION */}
      <section className="auth-form-section">

        <div className="auth-card">

          <div className="auth-logo">

            <div className="auth-logo-icon">
              <Activity size={24} />
            </div>

            <div>
              <span>MEDXAI</span>
              <small>Medical AI Platform</small>
            </div>

          </div>


          <div className="auth-header">

            <span className="auth-eyebrow">
              GET STARTED
            </span>

            <h2>Create Account</h2>

            <p>
              Choose your account type to get started.
            </p>

          </div>


          {/* ROLE SELECTOR */}
          <div className="role-selector">

            <button
              type="button"
              className={`role ${
                role === "doctor" ? "active" : ""
              }`}
              onClick={() => {
                setRole("doctor");
                setError("");
              }}
            >
              <Stethoscope size={20} />

              <span>
                <strong>Doctor</strong>
                <small>Clinical access</small>
              </span>

            </button>


            <button
              type="button"
              className={`role ${
                role === "patient" ? "active" : ""
              }`}
              onClick={() => {
                setRole("patient");
                setError("");
              }}
            >
              <UserRound size={20} />

              <span>
                <strong>Patient</strong>
                <small>Personal access</small>
              </span>

            </button>

          </div>


          {/* REGISTER FORM */}
          <form
            className="auth-form"
            onSubmit={handleSubmit}
          >

            <div className="auth-field">

              <label htmlFor="register-name">
                Full Name
              </label>

              <div className="input-with-icon">

                <UserRound
                  size={18}
                  className="input-icon"
                />

                <input
                  id="register-name"
                  type="text"
                  placeholder="Enter your full name"
                  value={name}
                  onChange={(e) =>
                    setName(e.target.value)
                  }
                  autoComplete="name"
                  required
                />

              </div>

            </div>


            <div className="auth-field">

              <label htmlFor="register-email">
                Email Address
              </label>

              <div className="input-with-icon">

                <Mail
                  size={18}
                  className="input-icon"
                />

                <input
                  id="register-email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) =>
                    setEmail(e.target.value)
                  }
                  autoComplete="email"
                  required
                />

              </div>

            </div>


            <div className="auth-field">

              <label htmlFor="register-password">
                Password
              </label>

              <div className="input-with-icon">

                <Lock
                  size={18}
                  className="input-icon"
                />

                <input
                  id="register-password"
                  type="password"
                  placeholder="Create a password"
                  value={password}
                  onChange={(e) =>
                    setPassword(e.target.value)
                  }
                  autoComplete="new-password"
                  minLength={6}
                  required
                />

              </div>

            </div>


            {error && (
              <div className="auth-error">
                {error}
              </div>
            )}


            <button
              type="submit"
              className="auth-submit-btn"
              disabled={loading}
            >
              {loading
                ? "Creating account..."
                : role === "doctor"
                ? "Create Doctor Account"
                : "Create Patient Account"}
            </button>

          </form>


          <div className="auth-security">

            <ShieldCheck size={16} />

            <span>
              Secure registration · Protected medical data
            </span>

          </div>


          <p className="auth-switch">
            Already have an account?
            <Link to="/login">
              Sign in
            </Link>
          </p>

        </div>

      </section>

    </div>
  );
}