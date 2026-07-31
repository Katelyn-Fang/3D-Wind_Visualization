const API_URL = "http://127.0.0.1:8000";

export async function fetchMlWind(sample, signal) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sample),
    signal,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`ML service ${response.status}: ${message}`);
  }
  return response.json();
}

