const MAX_REQUEST_BYTES = 20 * 1024 * 1024;

function json(data, status = 200) {
  return Response.json(data, { status, headers: { "Cache-Control": "no-store" } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/status" && request.method === "GET") {
      const configured = Boolean(env.MODEL_API_URL);
      return json({
        ready: configured,
        model: "Face Synthetics Classifier",
        error: configured ? null : "Inference is not configured. Set MODEL_API_URL to a compatible model API.",
        input_size: 200,
        labels: { 0: "Real", 1: "Synthetic" },
        task: "Detects synthetic images from the Face Synthetics dataset; it is not a general deepfake detector.",
      });
    }

    if (url.pathname === "/api/predict" && request.method === "POST") {
      if (!env.MODEL_API_URL) {
        return json({ error: "Inference is not configured. Set MODEL_API_URL to a compatible model API." }, 503);
      }
      const declaredLength = Number(request.headers.get("content-length") || 0);
      if (declaredLength > MAX_REQUEST_BYTES) {
        return json({ error: "Upload is too large. The limit is 20 MB per request." }, 413);
      }
      let form;
      try {
        form = await request.formData();
      } catch {
        return json({ error: "Expected multipart form data." }, 400);
      }
      const images = form.getAll("images").filter((value) => value instanceof File && value.name);
      if (!images.length) return json({ error: "Choose at least one image." }, 400);
      if (images.length > 20) return json({ error: "Upload at most 20 images per request." }, 400);
      const threshold = Number(form.get("threshold") ?? 0.5);
      if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
        return json({ error: "Threshold must be between 0 and 1." }, 400);
      }

      const upstreamForm = new FormData();
      for (const image of images) upstreamForm.append("images", image, image.name);
      upstreamForm.append("threshold", String(threshold));
      try {
        const headers = new Headers();
        if (env.MODEL_API_TOKEN) headers.set("Authorization", `Bearer ${env.MODEL_API_TOKEN}`);
        const upstream = await fetch(env.MODEL_API_URL, {
          method: "POST", headers, body: upstreamForm,
          signal: AbortSignal.timeout(120_000),
        });
        const contentType = upstream.headers.get("content-type") || "application/json";
        return new Response(upstream.body, {
          status: upstream.status,
          headers: { "Content-Type": contentType, "Cache-Control": "no-store" },
        });
      } catch {
        return json({ error: "Could not reach the model API." }, 502);
      }
    }

    if (url.pathname.startsWith("/api/")) return json({ error: "Not found." }, 404);
    return env.ASSETS.fetch(request);
  },
};
