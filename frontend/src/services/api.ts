
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function checkBackendStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/status`, {
      method: "GET",
    });

    return response.ok;
  } catch (error) {
    console.error("Backend connection failed:", error);
    return false;
  }
}
