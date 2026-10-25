
import { useEffect, useRef, useState } from "react";
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
    let plugin: ReturnType<typeof Spectrogram.create> | undefined;
    let objectUrl: string | undefined;

    async function renderSpectrogram() {
      setError("");

      try {
        const audioContext = new AudioContext();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        if (cancelled || !containerRef.current) {
          await audioContext.close();
          return;
        }

        const samples = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;

        // Build a playable WAV from the decoded samples for WaveSurfer.
        const wavBuffer = createWav(samples, sampleRate);
        objectUrl = URL.createObjectURL(wavBuffer);

        const WaveSurfer = (await import("wavesurfer.js")).default;

        if (cancelled || !containerRef.current) {
          URL.revokeObjectURL(objectUrl);
          await audioContext.close();
          return;
        }

        const wavesurfer = WaveSurfer.create({
          container: containerRef.current,
          height: 220,
          waveColor: "#7d8ce8",
          progressColor: "#5267d8",
          url: objectUrl,
          interact: false,
        });

        plugin = Spectrogram.create({
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

        await audioContext.close();

        return () => {
          wavesurfer.destroy();
        };
      } catch {
        if (!cancelled) {
          setError("Unable to decode this audio file for spectrogram display.");
        }
      }
    }

    const cleanupPromise = renderSpectrogram();

    return () => {
      cancelled = true;
      URL.revokeObjectURL(objectUrl ?? "");
      void cleanupPromise;
      plugin = undefined;
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
