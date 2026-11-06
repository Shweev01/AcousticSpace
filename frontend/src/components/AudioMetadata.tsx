
import { useEffect, useState } from "react";

interface AudioMetadataProps {
  file: File;
}

interface Metadata {
  duration: number;
  sampleRate: number;
  channels: number;
}

export default function AudioMetadata({ file }: AudioMetadataProps) {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let audioContext: AudioContext | null = null;

    async function readMetadata() {
      setMetadata(null);
      setError("");

      try {
        audioContext = new AudioContext();
        const buffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(buffer);

        if (!cancelled) {
          setMetadata({
            duration: audioBuffer.duration,
            sampleRate: audioBuffer.sampleRate,
            channels: audioBuffer.numberOfChannels,
          });
        }
      } catch {
        if (!cancelled) {
          setError("Could not read audio metadata.");
        }
      } finally {
        if (audioContext && audioContext.state !== "closed") {
          await audioContext.close();
        }
      }
    }

    void readMetadata();

    return () => {
      cancelled = true;

      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close();
      }
    };
  }, [file]);

  function formatDuration(seconds: number) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
  }

  return (
    <section className="metadata-card">
      <h3>Audio metadata</h3>

      {error ? (
        <p className="error-message">{error}</p>
      ) : !metadata ? (
        <p>Reading audio information...</p>
      ) : (
        <div className="metadata-grid">
          <div>
            <span>Duration</span>
            <strong>{formatDuration(metadata.duration)}</strong>
          </div>

          <div>
            <span>Sample rate</span>
            <strong>{metadata.sampleRate.toLocaleString()} Hz</strong>
          </div>

          <div>
            <span>Channels</span>
            <strong>{metadata.channels}</strong>
          </div>

          <div>
            <span>File format</span>
            <strong>{file.name.split(".").pop()?.toUpperCase() || "Unknown"}</strong>
          </div>
        </div>
      )}
    </section>
  );
}
