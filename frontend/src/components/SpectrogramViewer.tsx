
import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import Spectrogram from "wavesurfer.js/dist/plugins/spectrogram.esm.js";

type SpectrogramViewerProps = {
  file: File;
};

export default function SpectrogramViewer({
  file,
}: SpectrogramViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let wavesurfer: WaveSurfer | null = null;
    let audioContext: AudioContext | null = null;
    let objectUrl: string | null = null;

    async function renderSpectrogram() {
      setError("");

      try {
        audioContext = new AudioContext();

        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        if (cancelled || !containerRef.current) return;

        const samples = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;

        const wavBuffer = createWav(samples, sampleRate);
        objectUrl = URL.createObjectURL(wavBuffer);

        wavesurfer = WaveSurfer.create({
          container: containerRef.current,
          height: 220,
          waveColor: "#7d8ce8",
          progressColor: "#5267d8",
          url: objectUrl,
          interact: false,
        });

        const plugin = Spectrogram.create({
          labels: true,
          height: 220,
          splitChannels: false,
          scale: "mel",
          frequencyMax: 8000,
          frequencyMin: 0,
        });

        wavesurfer.registerPlugin(plugin);

        wavesurfer.on("error", () => {
          if (!cancelled) {
            setError("Could not render the spectrogram for this audio file.");
          }
        });
      } catch {
        if (!cancelled) {
          setError("Unable to decode this audio file for spectrogram display.");
        }
      } finally {
        if (audioContext && audioContext.state !== "closed") {
          await audioContext.close();
        }
      }
    }

    void renderSpectrogram();

    return () => {
      cancelled = true;
      wavesurfer?.destroy();

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }

      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close();
      }
    };
  }, [file]);

  return (
    <div className="spectrogram-viewer">
      {error ? (
        <p className="error-message" role="alert">{error}</p>
      ) : (
        <div ref={containerRef} className="spectrogram-canvas" />
      )}

      <p className="spectrogram-caption">
        Frequency content of the uploaded recording · visualisation only
      </p>
    </div>
  );
}

function createWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset: number, text: string) {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));

    view.setInt16(
      44 + i * 2,
      value < 0 ? value * 32768 : value * 32767,
      true
    );
  }

  return new Blob([buffer], { type: "audio/wav" });
}
