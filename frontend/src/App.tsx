import { useEffect, useRef, useState } from "react";
import WaveformViewer, {
  type WaveformViewerHandle,
} from "./components/WaveformViewer";
import SpectrogramViewer from "./components/SpectrogramViewer";
import AudioMetadata from "./components/AudioMetadata";
import AnalysisHistory, {
  type HistoryEntry,
} from "./components/AnalysisHistory";
import { checkBackendStatus } from "./services/api";
import "./App.css";

type AnalysisResult = {
  prediction: "Real" | "Fake";
  confidence: number;
  rt60: number;
  drr: number;
  segments: {
    start: number;
    end: number;
    score: number;
  }[];
};

const HISTORY_STORAGE_KEY = "acousticspace-analysis-history";

const demoResult: AnalysisResult = {
  prediction: "Fake",
  confidence: 92,
  rt60: 0.45,
  drr: 6.8,
  segments: [
    { start: 1.2, end: 2.4, score: 0.95 },
    { start: 4.1, end: 5.0, score: 0.78 },
  ],
};

function loadHistory(): HistoryEntry[] {
  try {
    const saved = localStorage.getItem(HISTORY_STORAGE_KEY);

    if (!saved) return [];

    const parsed: unknown = JSON.parse(saved);

    if (!Array.isArray(parsed)) return [];

    return parsed.filter(
      (entry): entry is HistoryEntry =>
        entry !== null &&
        typeof entry === "object" &&
        typeof entry.id === "string" &&
        typeof entry.filename === "string" &&
        typeof entry.timestamp === "string" &&
        (entry.prediction === "Real" || entry.prediction === "Fake") &&
        typeof entry.confidence === "number"
    );
  } catch {
    return [];
  }
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const waveformRef = useRef<WaveformViewerHandle>(null);

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null);

  // Backend connection status
  const [backendConnected, setBackendConnected] = useState(false);

  // Save analysis history in browser storage
  useEffect(() => {
    try {
      localStorage.setItem(
        HISTORY_STORAGE_KEY,
        JSON.stringify(history)
      );
    } catch {
      setError(
        "History could not be saved. Check your browser storage settings."
      );
    }
  }, [history]);

  // Check backend connection every 10 seconds
  useEffect(() => {
    let isMounted = true;

    const checkStatus = async () => {
      const connected = await checkBackendStatus();

      if (isMounted) {
        setBackendConnected(connected);
      }
    };

    checkStatus();

    const interval = window.setInterval(checkStatus, 10000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  function handleFile(file?: File) {
    setError("");

    if (!file) return;

    const valid = [".wav", ".flac"].some((extension) =>
      file.name.toLowerCase().endsWith(extension)
    );

    if (!valid) {
      setError("Please select a WAV or FLAC audio file.");
      return;
    }

    if (file.size === 0) {
      setError("This file is empty. Please select another file.");
      return;
    }

    setAudioFile(file);
    setResult(null);
    setSelectedSegment(null);
  }

  function resetAnalysis() {
    setAudioFile(null);
    setResult(null);
    setError("");
    setIsAnalyzing(false);
    setSelectedSegment(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function formatSize(bytes: number) {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function handleAnalyze() {
    if (!audioFile || isAnalyzing) return;

    setError("");
    setIsAnalyzing(true);

    // Demo only: no actual deepfake detection is performed.
    const demo = { ...demoResult };
    setResult(demo);
    setSelectedSegment(null);

    const newEntry: HistoryEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      filename: audioFile.name,
      timestamp: new Date().toISOString(),
      prediction: demo.prediction,
      confidence: demo.confidence,
    };

    setHistory((previous) => [newEntry, ...previous]);
    setIsAnalyzing(false);
  }

  function clearHistory() {
    setHistory([]);

    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      setError("Could not clear history from browser storage.");
    }
  }

  function selectSegment(index: number, start: number) {
    setSelectedSegment(index);
    waveformRef.current?.seekToTime(start);
  }

  function downloadReport() {
    if (!audioFile || !result) return;

    const report = {
      project: "AcousticSpace",
      reportType: "Demo Analysis Report",
      audioFile: audioFile.name,
      fileSize: formatSize(audioFile.size),
      analysisDate: new Date().toISOString(),
      isDemo: true,
      notice:
        "Illustrative mock results only. No actual deepfake detection or acoustic analysis was performed.",
      results: {
        prediction: result.prediction,
        confidence: `${result.confidence}%`,
        rt60Seconds: result.rt60,
        drrDecibels: result.drr,
        suspiciousSegments: result.segments.map((segment) => ({
          startSeconds: segment.start,
          endSeconds: segment.end,
          demoAnomalyScore: `${Math.round(segment.score * 100)}%`,
        })),
      },
    };

    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `AcousticSpace-Report-${Date.now()}.json`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    // Revoke after the browser has had time to start the download.
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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

          {/* Backend status and demo mode */}
          <div className="topbar-status-group">
            <div
              className={`backend-status ${
                backendConnected ? "connected" : "disconnected"
              }`}
            >
              <span className="backend-status-dot" />
              Backend {backendConnected ? "Connected" : "Disconnected"}
            </div>

            <div className="prototype-label">
              <span /> Demo mode
            </div>
          </div>
        </header>

        <section className="page-content">
          <div className="page-heading">
            <p className="eyebrow">AUDIO FORENSICS PLATFORM</p>
            <h1>Deepfake Audio Detection</h1>
            <p className="subtitle">
              Analyze audio characteristics and room acoustics to
              investigate potentially synthetic speech.
            </p>
          </div>

          <div className="info-banner">
            <span className="info-icon">i</span>
            <p>
              <strong>Demo mode — mock results</strong>
              <br />
              Results are illustrative and are not generated from
              your uploaded audio.
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
              <div className="upload-icon">⇧</div>
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

            {audioFile && (
              <>
                <WaveformViewer
                  ref={waveformRef}
                  file={audioFile}
                />
                <AudioMetadata file={audioFile} />
              </>
            )}

            <button
              type="button"
              className="analyze-button"
              disabled={!audioFile || isAnalyzing}
              onClick={handleAnalyze}
            >
              {isAnalyzing ? "Analyzing…" : "Run demo analysis"}
              <span>→</span>
            </button>

            <p className="privacy-note">
              Your audio stays in this browser; demo results are
              not calculated from the recording.
            </p>
          </section>

          <section className="next-section">
            <div className="section-heading">
              <div className="step-number">02</div>
              <div>
                <h2>Analysis results</h2>
                <p>Detection and acoustic indicators.</p>
              </div>
            </div>

            {!result ? (
              <div className="empty-results">
                <div className="wave-decoration">〰</div>
                <h3>Waiting for analysis</h3>
                <p>
                  Upload an audio file and run the demo to preview
                  the results layout.
                </p>
              </div>
            ) : (
              <div className="results-dashboard">
                <div className="results-toolbar">
                  <div className="demo-warning">
                    Illustrative mock data — not a real detection
                  </div>

                  <button
                    type="button"
                    className="download-report-button"
                    onClick={downloadReport}
                  >
                    ↓ Download Report
                  </button>
                </div>

                <div className="metrics-grid">
                  <article className="metric-card prediction-card">
                    <span className="metric-label">PREDICTION</span>
                    <h3>{result.prediction}</h3>
                    <p>Demo classification</p>
                  </article>

                  <article className="metric-card">
                    <span className="metric-label">CONFIDENCE</span>
                    <h3>{result.confidence}%</h3>
                    <div className="confidence-track">
                      <div
                        style={{ width: `${result.confidence}%` }}
                      />
                    </div>
                  </article>

                  <article className="metric-card">
                    <span className="metric-label">RT60</span>
                    <h3>
                      {result.rt60} <small>s</small>
                    </h3>
                    <p>Reverberation time</p>
                  </article>

                  <article className="metric-card">
                    <span className="metric-label">DRR</span>
                    <h3>
                      {result.drr} <small>dB</small>
                    </h3>
                    <p>Direct-to-reverberant ratio</p>
                  </article>
                </div>

                <article className="result-panel">
                  <h3>Suspicious audio segments</h3>
                  <p className="panel-description">
                    Demo intervals only; not measured from your recording.
                    Select a segment to jump to its start in the waveform.
                  </p>

                  <div className="segment-header">
                    <span>Time interval</span>
                    <span>Demo anomaly score</span>
                  </div>

                  {result.segments.map((segment, index) => (
                    <button
                      type="button"
                      className={`segment-row segment-button ${
                        selectedSegment === index
                          ? "segment-selected"
                          : ""
                      }`}
                      key={`${segment.start}-${index}`}
                      onClick={() =>
                        selectSegment(index, segment.start)
                      }
                      aria-pressed={selectedSegment === index}
                      title={`Jump to ${segment.start.toFixed(1)} seconds`}
                    >
                      <span>
                        {segment.start.toFixed(1)}s –{" "}
                        {segment.end.toFixed(1)}s
                      </span>

                      <div className="segment-score">
                        <div className="confidence-track">
                          <div
                            style={{
                              width: `${segment.score * 100}%`,
                            }}
                          />
                        </div>
                        <strong>
                          {Math.round(segment.score * 100)}%
                        </strong>
                      </div>

                      <span className="segment-action">
                        {selectedSegment === index
                          ? "Selected ✓"
                          : "▶ Jump to segment"}
                      </span>
                    </button>
                  ))}
                </article>

                <article className="result-panel">
                  <h3>Spectrogram</h3>
                  <p className="panel-description">
                    Frequency content of the uploaded audio. This
                    visualization does not determine whether audio
                    is real or fake.
                  </p>

                  {audioFile ? (
                    <SpectrogramViewer file={audioFile} />
                  ) : (
                    <p>Select an audio file to view its spectrogram.</p>
                  )}
                </article>
              </div>
            )}
          </section>

          <AnalysisHistory
            entries={history}
            onClear={clearHistory}
          />

          <footer>
            AcousticSpace <span>·</span> Research prototype for
            audio forensics
          </footer>
        </section>
      </main>
    </div>
  );
}

export default App;