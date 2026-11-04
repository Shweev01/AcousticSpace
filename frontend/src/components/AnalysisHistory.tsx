
export interface HistoryEntry {
  id: string;
  filename: string;
  timestamp: string;
  prediction: "Real" | "Fake";
  confidence: number;
}

interface AnalysisHistoryProps {
  entries: HistoryEntry[];
  onClear: () => void;
}

export default function AnalysisHistory({
  entries,
  onClear,
}: AnalysisHistoryProps) {
  return (
    <section className="history-card">
      <div className="history-heading">
        <div>
          <h2>Analysis history</h2>
          <p>Your previous demo analysis sessions.</p>
        </div>

        {entries.length > 0 && (
          <button
            type="button"
            className="clear-history-button"
            onClick={onClear}
          >
            Clear history
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="history-empty">
          <p>No analysis history yet.</p>
          <span>Your completed demo analyses will appear here.</span>
        </div>
      ) : (
        <div className="history-list">
          {entries.map((entry) => (
            <article className="history-item" key={entry.id}>
              <div className="history-file-icon">♫</div>

              <div className="history-file-details">
                <strong>{entry.filename}</strong>
                <span>
                  {new Date(entry.timestamp).toLocaleString()}
                </span>
              </div>

              <div className="history-result">
                <strong>{entry.prediction}</strong>
                <span>{entry.confidence}% confidence</span>
              </div>
            </article>
          ))}
        </div>
      )}

      <p className="history-demo-note">
        Note: These are saved demo results, not actual deepfake detections.
      </p>
    </section>
  );
}
