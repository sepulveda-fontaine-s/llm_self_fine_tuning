const API_BASE = "http://127.0.0.1:8010";

const $ = (id) => document.getElementById(id);

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function milliseconds(value) {
  return `${Number(value).toFixed(1)} ms`;
}

function uptime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }

  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }

  return `${secs}s`;
}

function setStatus(status) {
  const dot = $("status-dot");
  const label = $("service-status");

  dot.classList.remove("healthy", "unhealthy");

  if (status === "healthy") {
    dot.classList.add("healthy");
    label.textContent = "Healthy";
    label.className = "good";
  } else if (status === "degraded") {
    label.textContent = "Degraded";
    label.className = "warn";
  } else {
    dot.classList.add("unhealthy");
    label.textContent = "Unavailable";
    label.className = "bad";
  }
}

async function loadHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    setStatus(data.status);
  } catch (error) {
    setStatus("unhealthy");
  }
}

function drawLatencyChart(history) {
  const canvas = $("latency-chart");
  const context = canvas.getContext("2d");

  const width = canvas.width;
  const height = canvas.height;

  context.clearRect(0, 0, width, height);

  const values = history
    .map((point) => Number(point.end_to_end_latency_ms || 0))
    .slice(-40);

  context.strokeStyle = "rgba(158, 171, 200, 0.25)";
  context.lineWidth = 1;

  for (let i = 1; i <= 4; i += 1) {
    const y = (height / 5) * i;

    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  if (values.length === 0) {
    return;
  }

  const maximum = Math.max(...values, 1);
  const padding = 18;

  context.strokeStyle = "#7aa2ff";
  context.lineWidth = 3;
  context.beginPath();

  values.forEach((value, index) => {
    const x =
      padding +
      (index / Math.max(values.length - 1, 1)) *
        (width - padding * 2);

    const y =
      height -
      padding -
      (value / maximum) *
        (height - padding * 2);

    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });

  context.stroke();
}

async function loadMetrics() {
  try {
    const response = await fetch(`${API_BASE}/metrics`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    $("uptime").textContent = uptime(data.uptime_seconds);
    $("requests-total").textContent = data.requests_total;
    $("error-rate").textContent = percent(data.error_rate);
    $("rpm").textContent = Number(
      data.requests_per_minute
    ).toFixed(1);

    $("e2e-latency").textContent = milliseconds(
      data.end_to_end_latency.p95_ms
    );

    $("gpu-memory").textContent =
      `${Number(data.gpu.memory_allocated_gib).toFixed(2)} / ` +
      `${Number(data.gpu.memory_total_gib).toFixed(2)} GiB`;

    $("gpu-name").textContent =
      data.gpu.device_name || "GPU unavailable";

    $("gpu-util").textContent =
      `${Number(data.gpu.utilization_percent).toFixed(1)}%`;

    $("rolling-window").textContent =
      data.rolling_window_size;

    $("latency-retrieval").textContent = milliseconds(
      data.retrieval_latency.last_ms
    );

    $("latency-generation").textContent = milliseconds(
      data.generation_latency.last_ms
    );

    $("latency-grounding").textContent = milliseconds(
      data.grounding_latency.last_ms
    );

    $("latency-e2e").textContent = milliseconds(
      data.end_to_end_latency.last_ms
    );

    $("last-updated").textContent =
      `Updated ${new Date().toLocaleTimeString()}`;

    drawLatencyChart(data.history || []);
  } catch (error) {
    $("last-updated").textContent =
      "Metrics unavailable";
  }
}

async function loadBenchmark() {
  try {
    const response = await fetch(`${API_BASE}/benchmark`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    $("recall-1").textContent =
      percent(data.retrieval.recall_at_1);

    $("recall-3").textContent =
      percent(data.retrieval.recall_at_3);

    $("recall-5").textContent =
      percent(data.retrieval.recall_at_5);

    $("recall-10").textContent =
      percent(data.retrieval.recall_at_10);

    $("mrr-10").textContent =
      Number(data.retrieval.mrr_at_10).toFixed(3);

    $("base-f1").textContent =
      Number(data.base.qa.task_f1).toFixed(3);

    $("base-em").textContent =
      Number(data.base.qa.task_exact_match).toFixed(3);

    $("base-entailment").textContent =
      percent(data.base.grounding.nli_entailment_rate);

    $("base-hallucination").textContent =
      percent(
        data.base.grounding.nli_hallucination_proxy_rate
      );

    $("ssl-f1").textContent =
      Number(data.self_supervised.qa.task_f1).toFixed(3);

    $("ssl-em").textContent =
      Number(
        data.self_supervised.qa.task_exact_match
      ).toFixed(3);

    $("ssl-entailment").textContent =
      percent(
        data.self_supervised.grounding.nli_entailment_rate
      );

    $("ssl-hallucination").textContent =
      percent(
        data.self_supervised.grounding
          .nli_hallucination_proxy_rate
      );
  } catch (error) {
    console.error("Benchmark loading failed:", error);
  }
}

function styleGrounding(data) {
  const label = $("nli-label");
  const proxy = $("hallucination-proxy");

  label.className = "";
  proxy.className = "";

  if (data.nli_label === "entailment") {
    label.classList.add("good");
  } else if (data.nli_label === "contradiction") {
    label.classList.add("bad");
  } else {
    label.classList.add("warn");
  }

  if (data.hallucination_proxy) {
    proxy.classList.add("bad");
  } else {
    proxy.classList.add("good");
  }
}

async function runInference() {
  const button = $("ask-button");
  const state = $("inference-state");

  const question = $("question").value.trim();
  const model = $("model-select").value;

  if (!question) {
    state.textContent = "Enter a question.";
    return;
  }

  button.disabled = true;
  state.textContent = "Running retrieval, generation and grounding...";

  try {
    const response = await fetch(`${API_BASE}/answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        model,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    $("answer-text").textContent =
      data.answer || "No answer returned.";

    $("evidence-text").textContent =
      data.retrieved_document?.text ||
      "No retrieved evidence.";

    $("nli-label").textContent =
      data.grounding.nli_label;

    $("nli-entailment").textContent =
      percent(data.grounding.entailment_probability);

    $("nli-contradiction").textContent =
      percent(data.grounding.contradiction_probability);

    $("hallucination-proxy").textContent =
      data.grounding.hallucination_proxy
        ? "Detected"
        : "Not detected";

    styleGrounding(data.grounding);

    state.textContent =
      `Completed in ${milliseconds(
        data.trace.total_latency_ms
      )}`;

    await loadMetrics();
  } catch (error) {
    console.error(error);
    state.textContent =
      `Inference failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

$("ask-button").addEventListener(
  "click",
  runInference
);

$("question").addEventListener(
  "keydown",
  (event) => {
    if (event.key === "Enter") {
      runInference();
    }
  }
);

async function initialize() {
  await Promise.all([
    loadHealth(),
    loadMetrics(),
    loadBenchmark(),
  ]);
}

initialize();

setInterval(loadHealth, 5000);
setInterval(loadMetrics, 3000);