// The particle engine, twice: an ambient field behind the hero copy, and the full
// simulation lab in its own section — requests as particles, the policy as geometry.
//
// Every particle carries a screen score (its vertical lane) and a judge score. In
// cascade mode the screen's two draggable thresholds split the flow three ways at
// gate 1: confident-risky bursts, confident-clean exits, and only the amber corridor
// between them travels on to pay for the judge at gate 2. Everything the meters say —
// cost, latency, escalation share — is computed from the actual routing counts and the
// lab's own parameter sliders, not from a script. Escalated particles slow down in the
// corridor in proportion to the judge's latency, so the latency cost is visible in the
// motion itself.

import { mulberry32, gaussian, clamp, money, PRICES, el } from "./util.js";

const VERT = `
attribute vec2 a_pos;
attribute vec4 a_color;
attribute float a_size;
uniform vec2 u_resolution;
varying vec4 v_color;
void main() {
  vec2 clip = (a_pos / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = a_size;
  v_color = a_color;
}`;

const FRAG = `
precision mediump float;
varying vec4 v_color;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float soft = smoothstep(1.0, 0.55, d);
  gl_FragColor = vec4(v_color.rgb, v_color.a * soft);
}`;

const UNSAFE_SHARE = 0.25;
const DEEP_THRESHOLD = 0.5;
const FLOW = 0, ESCALATED = 1, CLEARED = 2, BURSTING = 3;

// Scenarios spell out the regimes the experiments found. Separability drives how far
// apart the safe and risky score distributions sit on the cheap screen.
export const SCENARIOS = {
  typical: {
    label: "Typical traffic",
    description: "Safe and risky mostly separate, with a real overlap in the middle. "
      + "A modest uncertain band escalates; most of the bill disappears.",
    separability: 0.55, band: [0.48, 0.72],
  },
  easy: {
    label: "Easy traffic (UnSmile-like)",
    description: "Cleanly separable — the screen is nearly always sure. Almost nothing "
      + "escalates, and the saving approaches the 98% we measured at matched accuracy.",
    separability: 0.95, band: [0.45, 0.65],
  },
  adversarial: {
    label: "Adversarial (ToxicChat-like)",
    description: "Heavy overlap — the screen genuinely cannot tell. Most traffic "
      + "escalates, and the honest saving is small. Cascades are not magic; this is "
      + "the regime where the judge must simply run.",
    separability: 0.15, band: [0.30, 0.82],
  },
};

function buildGL(canvas) {
  const gl = canvas.getContext("webgl", { antialias: false, alpha: true, premultipliedAlpha: false });
  if (!gl) return null;
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
    }
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "program link failed");
  }
  gl.useProgram(program);
  const uResolution = gl.getUniformLocation(program, "u_resolution");
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  const STRIDE = 7;
  for (const [name, size, offset] of [["a_pos", 2, 0], ["a_color", 4, 2], ["a_size", 1, 6]]) {
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, STRIDE * 4, offset * 4);
  }
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.clearColor(0, 0, 0, 0);
  return { gl, uResolution, STRIDE };
}

// ── the ambient hero field: pure drift, no mechanics ──────────────────────
export function initHero(reducedMotion) {
  const canvas = document.getElementById("hero-canvas");
  if (!canvas) return;
  const context = buildGL(canvas);
  if (!context) { canvas.style.display = "none"; return; }
  const { gl, uResolution, STRIDE } = context;

  const rng = mulberry32(477001);
  const COUNT = window.innerWidth < 768 ? 2500 : 6000;
  const vertexData = new Float32Array(COUNT * STRIDE);
  const particles = [];
  let W = 0, H = 0, dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.parentElement.clientWidth;
    H = canvas.parentElement.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  resize();
  window.addEventListener("resize", resize);

  for (let i = 0; i < COUNT; i += 1) {
    particles.push({
      x: rng() * (W || 1440), y: rng() * (H || 700),
      speed: 14 + rng() * 34, phase: rng() * Math.PI * 2,
      unsafe: rng() < UNSAFE_SHARE, alpha: 0.35 + rng() * 0.5,
    });
  }

  function draw(now) {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uResolution, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    let offset = 0;
    for (const p of particles) {
      const [r, g, b] = p.unsafe ? [0.85, 0.47, 0.31] : [0.32, 0.6, 0.52];
      vertexData[offset] = p.x * dpr;
      vertexData[offset + 1] = (p.y + Math.sin(now * 0.0011 + p.phase) * 4) * dpr;
      vertexData[offset + 2] = r; vertexData[offset + 3] = g; vertexData[offset + 4] = b;
      vertexData[offset + 5] = p.alpha;
      vertexData[offset + 6] = 3.2 * dpr;
      offset += STRIDE;
    }
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.POINTS, 0, particles.length);
  }

  if (reducedMotion) { draw(0); return; }

  let last = performance.now();
  let running = true;
  new IntersectionObserver((entries) => { running = entries[0].isIntersecting; }).observe(canvas);
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (running) {
      for (const p of particles) {
        p.x += p.speed * dt;
        if (p.x > W + 10) { p.x = -10; p.y = Math.random() * H; }
      }
      draw(now);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ── the simulation lab ────────────────────────────────────────────────────
export function initSimLab(reducedMotion) {
  const canvas = document.getElementById("sim-canvas");
  const overlay = document.getElementById("sim-overlay");
  const context = buildGL(canvas);
  if (!context) {
    document.getElementById("sim-stage").style.display = "none";
    document.getElementById("sim-fallback").style.display = "block";
    return;
  }
  const { gl, uResolution, STRIDE } = context;

  const rng = mulberry32(20260811);
  const COUNT = window.innerWidth < 768 ? 4000 : 10000;
  const vertexData = new Float32Array(COUNT * STRIDE);

  // ── adjustable parameters: the lab's whole point ────────
  const params = {
    screenCost: PRICES.cheap,      // $/call
    judgeCost: PRICES.deep,
    screenLatency: 1200,           // ms — our measured flash-lite
    judgeLatency: 2500,            // ms — frontier-judge class
    separability: SCENARIOS.typical.separability,
  };
  let W = 0, H = 0, dpr = 1;
  let mode = "cascade";
  let [warn, block] = SCENARIOS.typical.band;

  const distribution = () => {
    const s = params.separability;
    return {
      safe: [clamp(0.47 - 0.32 * s, 0.05, 0.5), 0.2 - 0.1 * s],
      unsafe: [clamp(0.53 + 0.32 * s, 0.5, 0.95), 0.2 - 0.1 * s],
    };
  };

  const particles = [];
  const decisions = [];
  const DECISION_WINDOW = 700;

  const geometry = {
    padTop: 30, padBottom: 40,
    gate1: () => W * 0.42,
    gate2: () => W * 0.78,
    scoreToY: (s) => geometry.padTop + (1 - s) * (H - geometry.padTop - geometry.padBottom),
    yToScore: (y) => 1 - (y - geometry.padTop) / (H - geometry.padTop - geometry.padBottom),
  };

  function sampleScores() {
    const d = distribution();
    const unsafe = rng() < UNSAFE_SHARE;
    const [mean, sd] = unsafe ? d.unsafe : d.safe;
    return {
      unsafe,
      cheap: clamp(mean + gaussian(rng) * sd, 0.02, 0.98),
      deep: clamp((unsafe ? 0.8 : 0.22) + gaussian(rng) * 0.09, 0.02, 0.98),
    };
  }

  // Escalated particles crawl: their pace is the judge's latency, made visible.
  const escalatedFactor = () => clamp(1500 / params.judgeLatency, 0.25, 1.2);

  function spawn(particle, atLeft) {
    const s = sampleScores();
    particle.unsafe = s.unsafe;
    particle.cheap = s.cheap;
    particle.deep = s.deep;
    particle.x = atLeft ? -10 - rng() * 80 : rng() * W;
    particle.speed = 80 + rng() * 70;
    particle.phase = rng() * Math.PI * 2;
    particle.state = FLOW;
    particle.burst = 0;
    particle.alpha = 0.65 + rng() * 0.3;
    if (!atLeft) {
      if (mode === "cascade" && particle.x >= geometry.gate1()) {
        const verdict = routeAtScreen(particle.cheap);
        if (verdict === 2) { spawn(particle, true); return; }
        particle.state = verdict === 1 ? ESCALATED : CLEARED;
        if (particle.state === ESCALATED && particle.x >= geometry.gate2()) {
          if (particle.deep >= DEEP_THRESHOLD) { spawn(particle, true); return; }
          particle.state = CLEARED;
        }
      }
      if (mode === "flat" && particle.x >= geometry.gate2()) {
        if (particle.deep >= DEEP_THRESHOLD) { spawn(particle, true); return; }
        particle.state = CLEARED;
      }
    }
  }

  function routeAtScreen(cheap) {
    if (cheap >= block) return 2;
    if (cheap < warn) return 0;
    return 1;
  }

  function recordDecision(kind) {
    decisions.push(kind);
    if (decisions.length > DECISION_WINDOW) decisions.shift();
  }

  //! Seed the window by routing a fresh sample from the CURRENT distribution, so the
  //  meters answer immediately and re-settle the moment a slider moves. Same routing
  //  rule the particles obey — this is the share of traffic that escalates, computed
  //  the same way, not a placeholder.
  function primeDecisions(count = 400) {
    decisions.length = 0;
    for (let i = 0; i < count; i += 1) recordDecision(routeAtScreen(sampleScores().cheap));
  }

  function step(dt, now) {
    const gate1 = geometry.gate1();
    const gate2 = geometry.gate2();
    for (const p of particles) {
      if (p.state === BURSTING) {
        p.burst += dt * 3.2;
        if (p.burst >= 1) spawn(p, true);
        continue;
      }
      const factor = p.state === CLEARED ? 1.35 : p.state === ESCALATED ? escalatedFactor() : 1;
      p.x += p.speed * dt * factor;
      if (mode === "cascade" && p.state === FLOW && p.x >= gate1) {
        const verdict = routeAtScreen(p.cheap);
        recordDecision(verdict);
        if (verdict === 2) { p.state = BURSTING; p.burst = 0; p.x = gate1; }
        else p.state = verdict === 1 ? ESCALATED : CLEARED;
      }
      if (p.state === (mode === "cascade" ? ESCALATED : FLOW) && p.x >= gate2) {
        if (p.deep >= DEEP_THRESHOLD) { p.state = BURSTING; p.burst = 0; p.x = gate2; }
        else p.state = CLEARED;
      }
      if (p.x > W + 20) spawn(p, true);
      p.y = geometry.scoreToY(p.cheap) + Math.sin(now * 0.0016 + p.phase) * 2.5;
    }
  }

  function draw() {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uResolution, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    let offset = 0;
    for (const p of particles) {
      let r, g, b, a = p.alpha, size = 3.1 * dpr;
      if (p.unsafe) { r = 0.88; g = 0.48; b = 0.31; } else { r = 0.32; g = 0.62; b = 0.54; }
      if (p.state === ESCALATED) { r = 0.9; g = 0.72; b = 0.38; a = Math.min(1, a + 0.2); size = 3.5 * dpr; }
      if (p.state === CLEARED) { a *= 0.85; }
      if (p.state === BURSTING) {
        const t = p.burst;
        r = 0.92; g = 0.36; b = 0.28;
        a = (1 - t) * 0.95;
        size = (3.1 + t * 8) * dpr;
      }
      vertexData[offset] = p.x * dpr;
      vertexData[offset + 1] = p.y * dpr;
      vertexData[offset + 2] = r; vertexData[offset + 3] = g; vertexData[offset + 4] = b;
      vertexData[offset + 5] = a;
      vertexData[offset + 6] = size;
      offset += STRIDE;
    }
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.POINTS, 0, particles.length);
  }

  // ── overlay ─────────────────────────────────────────────
  const nodes = {};
  function buildOverlay() {
    overlay.textContent = "";
    nodes.corridor = el("rect", { class: "corridor", rx: 3 }, overlay);
    nodes.gate1 = el("line", { class: "gate" }, overlay);
    nodes.gate2 = el("line", { class: "gate deep" }, overlay);
    nodes.gate1Label = el("text", {}, overlay);
    nodes.gate2Label = el("text", {}, overlay);
    nodes.zoneBlock = el("text", { class: "zone" }, overlay);
    nodes.zoneClear = el("text", { class: "zone" }, overlay);
    nodes.zoneEsc = el("text", { class: "zone" }, overlay);
    for (const which of ["warn", "block"]) {
      const group = el("g", { class: "handle", tabindex: 0, role: "slider",
        "aria-orientation": "vertical", "aria-label": which === "warn"
          ? "escalation threshold: below this, requests exit without the judge"
          : "blocking threshold: at or above this, requests are blocked by the screen" }, overlay);
      el("circle", { class: "halo", r: 15 }, group);
      el("circle", { class: "grip", r: 6 }, group);
      const label = el("text", { x: 22, y: 4 }, group);
      nodes[which] = { group, label };
      attachDrag(group, which);
    }
  }

  function layoutOverlay() {
    const gate1 = geometry.gate1(), gate2 = geometry.gate2();
    const yWarn = geometry.scoreToY(warn), yBlock = geometry.scoreToY(block);
    const cascade = mode === "cascade";

    nodes.gate1.setAttribute("x1", gate1); nodes.gate1.setAttribute("x2", gate1);
    nodes.gate1.setAttribute("y1", geometry.padTop); nodes.gate1.setAttribute("y2", H - geometry.padBottom);
    nodes.gate1.style.display = cascade ? "" : "none";
    nodes.gate1Label.style.display = cascade ? "" : "none";
    nodes.gate1Label.setAttribute("x", gate1 - 6);
    nodes.gate1Label.setAttribute("y", geometry.padTop + 4);
    nodes.gate1Label.setAttribute("text-anchor", "end");
    nodes.gate1Label.textContent =
      `screen · ${money(params.screenCost * 1000, 3)}/1k · ${Math.round(params.screenLatency)}ms`;

    const g2Top = cascade ? yBlock : geometry.padTop;
    const g2Bottom = cascade ? yWarn : H - geometry.padBottom;
    nodes.gate2.setAttribute("x1", gate2); nodes.gate2.setAttribute("x2", gate2);
    nodes.gate2.setAttribute("y1", g2Top); nodes.gate2.setAttribute("y2", g2Bottom);
    nodes.gate2Label.setAttribute("x", gate2 - 6);
    nodes.gate2Label.setAttribute("text-anchor", "end");
    nodes.gate2Label.setAttribute("y", geometry.padTop + 4);
    nodes.gate2Label.textContent =
      `judge · ${money(params.judgeCost * 1000, 2)}/1k · ${Math.round(params.judgeLatency)}ms`;

    nodes.corridor.style.display = cascade ? "" : "none";
    nodes.corridor.setAttribute("x", gate1);
    nodes.corridor.setAttribute("y", yBlock);
    nodes.corridor.setAttribute("width", Math.max(0, gate2 - gate1));
    nodes.corridor.setAttribute("height", Math.max(0, yWarn - yBlock));

    for (const [which, score] of [["warn", warn], ["block", block]]) {
      const handle = nodes[which];
      handle.group.style.display = cascade ? "" : "none";
      handle.group.setAttribute("transform", `translate(${gate1}, ${geometry.scoreToY(score)})`);
      handle.group.setAttribute("aria-valuenow", score.toFixed(2));
      handle.label.textContent = which === "warn" ? `escalate ≥ ${warn.toFixed(2)}` : `block ≥ ${block.toFixed(2)}`;
    }

    // Captions hug the right end of the corridor, end-anchored, so they never
    // collide with the threshold handles and their labels on the left.
    const zoneX = gate2 - 8;
    for (const node of [nodes.zoneBlock, nodes.zoneClear, nodes.zoneEsc]) {
      node.style.display = cascade ? "" : "none";
      node.setAttribute("x", zoneX);
      node.setAttribute("text-anchor", "end");
    }
    nodes.zoneBlock.setAttribute("y", Math.max(geometry.padTop + 34, yBlock - 14));
    nodes.zoneBlock.textContent = "BLOCKED AT THE SCREEN";
    nodes.zoneEsc.setAttribute("y", (yWarn + yBlock) / 2 + 3);
    nodes.zoneEsc.textContent = "UNCERTAIN → PAYS THE JUDGE";
    nodes.zoneClear.setAttribute("y", Math.min(H - geometry.padBottom - 10, yWarn + 24));
    nodes.zoneClear.textContent = "CLEARED — NEVER PAYS, NEVER WAITS";
  }

  function attachDrag(group, which) {
    let dragging = false;
    group.addEventListener("pointerdown", (event) => {
      dragging = true; group.setPointerCapture(event.pointerId); event.preventDefault();
    });
    group.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const rect = overlay.getBoundingClientRect();
      setThreshold(which, geometry.yToScore(event.clientY - rect.top));
    });
    group.addEventListener("pointerup", () => { dragging = false; });
    group.addEventListener("keydown", (event) => {
      const delta = event.key === "ArrowUp" ? 0.02 : event.key === "ArrowDown" ? -0.02 : 0;
      if (!delta) return;
      event.preventDefault();
      setThreshold(which, (which === "warn" ? warn : block) + delta);
    });
  }

  function setThreshold(which, value) {
    if (which === "warn") warn = clamp(value, 0.04, block - 0.03);
    else block = clamp(value, warn + 0.03, 0.96);
    primeDecisions();
    layoutOverlay();
    if (reducedMotion) renderStatic();
  }

  // ── meters: cost AND latency, honestly ──────────────────
  const meter = (id) => document.getElementById(id);
  const meters = {
    costFlat: meter("sim-cost-flat"), costCascade: meter("sim-cost-cascade"),
    costSave: meter("sim-cost-save"), esc: meter("sim-esc"),
    latFlat: meter("sim-lat-flat"), latMean: meter("sim-lat-mean"),
    latSave: meter("sim-lat-save"), latP95: meter("sim-lat-p95"),
  };

  function updateMeters() {
    meters.costFlat.textContent = money(params.judgeCost * 1000, 2);
    meters.latFlat.textContent = Math.round(params.judgeLatency) + "ms";
    if (mode === "flat" || decisions.length < 60) {
      if (mode === "flat") {
        for (const key of ["costCascade", "costSave", "esc", "latMean", "latSave", "latP95"]) {
          meters[key].textContent = "—";
        }
      }
      return;
    }
    let escalated = 0;
    for (const d of decisions) if (d === 1) escalated += 1;
    const share = escalated / decisions.length;

    const costCascade = 1000 * (params.screenCost + share * params.judgeCost);
    const costFlat = 1000 * params.judgeCost;
    meters.costCascade.textContent = money(costCascade, 2);
    meters.costSave.textContent = "−" + ((1 - costCascade / costFlat) * 100).toFixed(1) + "%";
    meters.esc.textContent = (share * 100).toFixed(1) + "%";

    // Mean latency collapses; the p95 is the honest tail — an escalated request
    // waits for the screen AND the judge, longer than judge-everything would take.
    const latMean = params.screenLatency + share * params.judgeLatency;
    const latP95 = share >= 0.05
      ? params.screenLatency + params.judgeLatency
      : params.screenLatency;
    meters.latMean.textContent = Math.round(latMean) + "ms";
    meters.latSave.textContent = (latMean <= params.judgeLatency ? "−" : "+")
      + Math.abs((1 - latMean / params.judgeLatency) * 100).toFixed(0) + "%";
    meters.latP95.textContent = Math.round(latP95) + "ms"
      + (latP95 > params.judgeLatency ? " ⚠ tail waits for both" : "");
  }

  // ── controls: mode, scenarios, sliders ──────────────────
  const modeFlat = document.getElementById("sim-mode-flat");
  const modeCascade = document.getElementById("sim-mode-cascade");
  function setMode(next) {
    mode = next;
    modeFlat.setAttribute("aria-pressed", String(mode === "flat"));
    modeCascade.setAttribute("aria-pressed", String(mode === "cascade"));
    primeDecisions();
    for (const p of particles) {
      if (p.state !== BURSTING && p.x < (mode === "cascade" ? geometry.gate1() : geometry.gate2())) {
        p.state = FLOW;
      }
    }
    layoutOverlay();
    updateMeters();
    if (reducedMotion) renderStatic();
  }
  modeFlat.addEventListener("click", () => setMode("flat"));
  modeCascade.addEventListener("click", () => setMode("cascade"));

  function reshuffleUndecided() {
    primeDecisions();
    for (const p of particles) {
      if (p.x < geometry.gate1() || p.state === BURSTING) {
        const s = sampleScores();
        p.unsafe = s.unsafe; p.cheap = s.cheap; p.deep = s.deep;
        if (p.state !== BURSTING) p.state = FLOW;
      }
    }
  }

  for (const [key, scenario] of Object.entries(SCENARIOS)) {
    const input = document.getElementById(`scenario-${key}`);
    input.addEventListener("change", () => {
      params.separability = scenario.separability;
      [warn, block] = scenario.band;
      syncSliders();
      reshuffleUndecided();
      if (mode !== "cascade") setMode("cascade");
      layoutOverlay();
      if (reducedMotion) renderStatic();
    });
  }

  // Sliders: costs are log-scaled (they span decades); latencies and separability
  // are linear. Every change re-labels the gates and reprices the meters live.
  const sliders = [
    { id: "sim-screen-cost", key: "screenCost", log: [0.000001, 0.001],
      format: (v) => money(v * 1000, 3) + "/1k" },
    { id: "sim-judge-cost", key: "judgeCost", log: [0.0001, 0.01],
      format: (v) => money(v * 1000, 2) + "/1k" },
    { id: "sim-screen-lat", key: "screenLatency", linear: [5, 3000],
      format: (v) => Math.round(v) + "ms" },
    { id: "sim-judge-lat", key: "judgeLatency", linear: [200, 8000],
      format: (v) => Math.round(v) + "ms" },
    { id: "sim-sep", key: "separability", linear: [0, 1],
      format: (v) => v < 0.33 ? "overlapping" : v < 0.7 ? "partly separable" : "cleanly separable" },
  ];
  function sliderToValue(s, t) {
    if (s.log) {
      const [lo, hi] = s.log;
      return lo * Math.pow(hi / lo, t);
    }
    const [lo, hi] = s.linear;
    return lo + (hi - lo) * t;
  }
  function valueToSlider(s, v) {
    if (s.log) {
      const [lo, hi] = s.log;
      return Math.log(v / lo) / Math.log(hi / lo);
    }
    const [lo, hi] = s.linear;
    return (v - lo) / (hi - lo);
  }
  function syncSliders() {
    for (const s of sliders) {
      const input = document.getElementById(s.id);
      input.value = String(Math.round(valueToSlider(s, params[s.key]) * 1000));
      document.getElementById(s.id + "-value").textContent = s.format(params[s.key]);
    }
  }
  for (const s of sliders) {
    const input = document.getElementById(s.id);
    input.addEventListener("input", () => {
      params[s.key] = sliderToValue(s, Number(input.value) / 1000);
      document.getElementById(s.id + "-value").textContent = s.format(params[s.key]);
      if (s.key === "separability") {
        for (const key of Object.keys(SCENARIOS)) {
          document.getElementById(`scenario-${key}`).checked = false;
        }
        reshuffleUndecided();
      }
      primeDecisions();
      layoutOverlay();
      updateMeters();
      if (reducedMotion) renderStatic();
    });
  }

  // ── lifecycle ───────────────────────────────────────────
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const stage = canvas.parentElement;
    W = stage.clientWidth;
    H = stage.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    overlay.setAttribute("width", W);
    overlay.setAttribute("height", H);
    overlay.setAttribute("viewBox", `0 0 ${W} ${H}`);
    layoutOverlay();
  }

  function renderStatic() {
    primeDecisions();
    for (let i = 0; i < 420; i += 1) step(1 / 30, i * 33);
    draw();
    updateMeters();
  }

  buildOverlay();
  resize();
  for (let i = 0; i < COUNT; i += 1) {
    const p = {};
    particles.push(p);
    spawn(p, false);
  }
  syncSliders();
  primeDecisions();
  updateMeters();
  document.getElementById("scenario-typical").checked = true;
  window.addEventListener("resize", resize);

  if (reducedMotion) { renderStatic(); return; }

  let last = performance.now();
  let meterClock = 0;
  let running = true;
  new IntersectionObserver((entries) => { running = entries[0].isIntersecting; }).observe(canvas);
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (running) {
      step(dt, now);
      draw();
      meterClock += dt;
      if (meterClock > 0.25) { meterClock = 0; updateMeters(); }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
