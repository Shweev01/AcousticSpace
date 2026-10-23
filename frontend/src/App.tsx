
import { useRef, useState } from "react";
import WaveformViewer from "./components/WaveformViewer";
import "./App.css";

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [error, setError] = useState("");

  function handleFile(file?: File) {
    setError("");

    if (!file) return;

    const validExtensions = [".wav", ".flac"];
    const isValid = validExtensions.some((extension) =>
      file.name.toLowerCase().endsWith(extension)
    );

    if (!isValid) {
      setError("Please select a WAV or FLAC audio file.");
      return;
    }

    if (file.size === 0) {
      setError("This file is empty. Please select another audio file.");
      return;
    }

    setAudioFile(file);
  }

  function formatSize(bytes: number) {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function resetAnalysis() {
    setAudioFile(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleAnalyze() {
    setError(
      "The backend is not connected yet. No deepfake detection has been performed."
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">A</div>
          <div>
            <h2>AcousticSpace</h2>
            <span>Audio Forensics</span>
          </div>
        </div>

        <p className="nav-heading">WORKSPACE</p>

        <button type="button" className="nav-item active">
          <span>◫</span> Audio Analysis
        </button>

        <button
          type="button"
          className="nav-item"
          onClick={resetAnalysis}
        >
          <span>◷</span> New Analysis
        </button>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>Project workspace</strong>
            <p>AcousticSpace v1.0</p>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> Audio Analysis
          </div>

          <div className="prototype-label">
            <span /> Prototype
          </div>
        </header>

        <section className="page-content">
          <div className="page-heading">
            <p className="eyebrow">AUDIO FORENSICS PLATFORM</p>
            <h1>Deepfake Audio Detection</h1>
            <p className="subtitle">
              Analyze audio characteristics and room acoustics
              to investigate potentially synthetic speech.
            </p>
          </div>

          <div className="info-banner">
            <span className="info-icon">i</span>
            <p>
              <strong>Research prototype</strong>
              <br />
              Detection results will become available after the
              audio-processing pipeline and ML model are integrated.
            </p>
          </div>

          <section className="upload-card">
            <div className="section-heading">
              <div className="step-number">01</div>
              <div>
                <h2>Upload audio</h2>
                <p>Select a recording to begin your analysis.</p>
              </div>
            </div>

            <div
              className="dropzone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleFile(event.dataTransfer.files[0]);
              }}
            >
              <div className="upload-icon">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 16V4" />
                  <path d="m7 9 5-5 5 5" />
                  <path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5" />
                </svg>
              </div>

              <h3>Drop your audio file here</h3>
              <p>or browse files from your computer</p>

              <button
                type="button"
                className="primary-button"
                onClick={() => fileInputRef.current?.click()}
              >
                Browse audio files
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.flac,audio/wav,audio/flac"
                hidden
                onChange={(event) => {
                  handleFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />

              <div className="file-guidelines">
                <span>WAV</span>
                <span>FLAC</span>
                <span>Audio files only</span>
              </div>
            </div>

            {error && (
              <p className="error-message" role="alert">
                {error}
              </p>
            )}

            {audioFile && (
              <div className="selected-file">
                <div className="file-symbol">♫</div>

                <div className="file-details">
                  <strong>{audioFile.name}</strong>
                  <span>{formatSize(audioFile.size)}</span>
                </div>

                <button
                  type="button"
                  className="remove-button"
                  onClick={resetAnalysis}
                  aria-label="Remove selected audio"
                >
                  ×
                </button>
              </div>
            )}

            {audioFile && <WaveformViewer file={audioFile} />}

            <button
              type="button"
              className="analyze-button"
              disabled={!audioFile}
              onClick={handleAnalyze}
            >
              Analyze audio <span>→</span>
            </button>

            <p className="privacy-note">
              Your recording is not uploaded to a server in this prototype.
            </p>
          </section>

          <section className="next-section">
            <div className="section-heading">
              <div className="step-number">02</div>
              <div>
                <h2>Analysis results</h2>
                <p>Your audio findings will appear here.</p>
              </div>
            </div>

            <div className="empty-results">
              <div className="wave-decoration">〰</div>
              <h3>Waiting for audio analysis</h3>
              <p>
                The waveform, acoustic features, spectrogram and
                prediction will appear after analysis is implemented.
              </p>
            </div>
          </section>

          <footer>
            AcousticSpace <span>·</span> Research prototype for audio forensics
          </footer>
        </section>
      </main>
    </div>
  );
}

export default App;
