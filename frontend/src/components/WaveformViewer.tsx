
import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

interface WaveformViewerProps {
  file: File;
}

function WaveformViewer({ file }: WaveformViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!containerRef.current) return;

    setError("");
    setIsPlaying(false);
    setDuration(0);
    setCurrentTime(0);

    const objectUrl = URL.createObjectURL(file);

    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      height: 76,
      waveColor: "#c5d2f8",
      progressColor: "#5273e8",
      cursorColor: "#243650",
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      normalize: true,
      url: objectUrl,
    });

    wavesurferRef.current = wavesurfer;

    wavesurfer.on("ready", (audioDuration) => {
      setDuration(audioDuration);
    });

    wavesurfer.on("timeupdate", (time) => {
      setCurrentTime(time);
    });

    wavesurfer.on("play", () => setIsPlaying(true));
    wavesurfer.on("pause", () => setIsPlaying(false));
    wavesurfer.on("finish", () => setIsPlaying(false));

    wavesurfer.on("error", () => {
      setError("Unable to decode this audio file. Try another WAV or FLAC file.");
    });

    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  function formatTime(seconds: number) {
    if (!Number.isFinite(seconds)) return "0:00";

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
  }

  async function togglePlayback() {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer || error) return;

    try {
      await wavesurfer.playPause();
    } catch {
      setError("Playback failed. Please try another audio file.");
    }
  }

  return (
    <section className="waveform-card">
      <div className="waveform-heading">
        <div>
          <h3>Audio waveform</h3>
          <p>Click the waveform to jump to a position.</p>
        </div>

        <span className="audio-badge">AUDIO LOADED</span>
      </div>

      <div className="waveform-container" ref={containerRef} />

      {error && <p className="error-message">{error}</p>}

      <div className="waveform-controls">
        <button
          className="play-button"
          onClick={togglePlayback}
          disabled={Boolean(error) || duration === 0}
          aria-label={isPlaying ? "Pause audio" : "Play audio"}
        >
          {isPlaying ? "Ⅱ" : "▶"}
        </button>

        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <span className="waveform-hint">Select waveform to seek</span>
      </div>
    </section>
  );
}

export default WaveformViewer;
