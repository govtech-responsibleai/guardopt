// A1 — the swarm. Requests as particles; the policy as geometry.
//
// Every particle carries a cheap-screen score (its vertical lane) and a deep-judge
// score. In cascade mode the screen's two draggable thresholds split the flow three
// ways at gate 1: confident-risky bursts, confident-clean exits, and only the amber
// corridor between them travels on to pay for the judge at gate 2. In "status quo"
// mode there is no screen: everything pays for the judge. The odometers price both,
// live, from the actual routing counts — not from a script.

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

// Traffic presets: the distributions ARE the story. Easy traffic separates cleanly
// (the UnSmile-like case: almost nothing escalates, the bill collapses); adversarial
// traffic overlaps (the ToxicChat-like case: the judge must run, and the honest
// number for the saving is small).
const TRAFFIC = {
  default: { safe: [0.28, 0.13], unsafe: [0.74, 0.13], band: [0.48, 0.72] },
  easy: { safe: [0.2, 0.08], unsafe: [0.85, 0.07], band: [0.45, 0.65] },
  adversarial: { safe: [0.42, 0.16], unsafe: [0.6, 0.16], band: [0.3, 0.82] },
};

const UNSAFE_SHARE = 0.25;
const DEEP_THRESHOLD = 0.5;

// states
const FLOW = 0, ESCALATED = 1, CLEARED = 2, BURSTING = 3;

export function initSwarm(reducedMotion) {
  const canvas = document.getElementById("swarm-canvas");
  const overlay = document.getElementById("swarm-overlay");
  const gl = canvas.getContext("webgl", { antialias: false, alpha: true, premultipliedAlpha: false });
  if (!gl) {
    canvas.style.display = "none";
    overlay.style.display = "none";
    document.getElementById("swarm-fallback").style.display = "block";
    return;
  }

  // ── GL setup ────────────────────────────────────────────
  const program = buildProgram(gl);
  gl.useProgram(program);
  const uResolution = gl.getUniformLocation(program, "u_resolution");
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const STRIDE = 7; // x y r g b a size
  const aPos = gl.getAttribLocation(program, "a_pos");
  const aColor = gl.getAttribLocation(program, "a_color");
  const aSize = gl.getAttribLocation(program, "a_size");
  gl.enableVertexAttribArray(aPos);
  gl.enableVertexAttribArray(aColor);
  gl.enableVertexAttribArray(aSize);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, STRIDE * 4, 0);
  gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, STRIDE * 4, 2 * 4);
  gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, STRIDE * 4, 6 * 4);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive: particles glow on the dark ground
  gl.clearColor(0, 0, 0, 0);

  // ── state ───────────────────────────────────────────────
  const rng = mulberry32(20260810);
  const mobile = window.innerWidth < 768;
  const COUNT = mobile ? 4500 : 12000;
  const vertexData = new Float32Array(COUNT * STRIDE);

  let W = 0, H = 0, dpr = 1;
  let mode = "cascade"; // or "flat"
  let traffic = TRAFFIC.default;
  let warn = traffic.band[0];
  let block = traffic.band[1];

  const particles = [];
  const decisions = []; // ring buffer of gate-1 outcomes: 0 exit, 1 escalate, 2 block
  const DECISION_WINDOW = 700;

  const geometry = {
    padTop: 30,
    padBottom: 40,
    gate1: () => W * 0.42,
    gate2: () => W * 0.76,
    scoreToY: (s) => geometry.padTop + (1 - s) * (H - geometry.padTop - geometry.padBottom),
    yToScore: (y) => 1 - (y - geometry.padTop) / (H - geometry.padTop - geometry.padBottom),
  };

  function sampleScores() {
    const unsafe = rng() < UNSAFE_SHARE;
    const [cm, cs] = unsafe ? traffic.unsafe : traffic.safe;
    const cheap = clamp(cm + gaussian(rng) * cs, 0.02, 0.98);
    const deep = clamp((unsafe ? 0.8 : 0.22) + gaussian(rng) * 0.09, 0.02, 0.98);
    return { unsafe, cheap, deep };
  }

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
    // Particles initialised mid-scene get their routing applied silently, so the
    // opening frame already looks like a running system rather than a starting gun.
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
    if (cheap >= block) return 2; // blocked
    if (cheap < warn) return 0;   // cleared — never pays the judge
    return 1;                     // escalate
  }

  function recordDecision(kind) {
    decisions.push(kind);
    if (decisions.length > DECISION_WINDOW) decisions.shift();
  }

  // ── simulation ──────────────────────────────────────────
  function step(dt, now) {
    const gate1 = geometry.gate1();
    const gate2 = geometry.gate2();
    for (const p of particles) {
      if (p.state === BURSTING) {
        p.burst += dt * 3.2;
        if (p.burst >= 1) spawn(p, true);
        continue;
      }
      p.x += p.speed * dt * (p.state === CLEARED ? 1.35 : 1);
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

  function fill() {
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
      vertexData[offset + 2] = r;
      vertexData[offset + 3] = g;
      vertexData[offset + 4] = b;
      vertexData[offset + 5] = a;
      vertexData[offset + 6] = size;
      offset += STRIDE;
    }
  }

  function draw() {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uResolution, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    fill();
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.POINTS, 0, particles.length);
  }

  // ── overlay: gates, corridor, handles, labels ───────────
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
    // Price labels sit at the top edge, end-anchored against their gate lines —
    // clear of the hero copy, the HUD meters, and the right edge at any width.
    nodes.gate1Label.setAttribute("x", gate1 - 6);
    nodes.gate1Label.setAttribute("y", geometry.padTop + 16);
    nodes.gate1Label.setAttribute("text-anchor", "end");
    nodes.gate1Label.textContent = `screen · ${money(PRICES.cheap * 1000, 3)}/1k`;

    const g2Top = cascade ? yBlock : geometry.padTop;
    const g2Bottom = cascade ? yWarn : H - geometry.padBottom;
    nodes.gate2.setAttribute("x1", gate2); nodes.gate2.setAttribute("x2", gate2);
    nodes.gate2.setAttribute("y1", g2Top); nodes.gate2.setAttribute("y2", g2Bottom);
    nodes.gate2Label.setAttribute("x", gate2 - 6);
    nodes.gate2Label.setAttribute("text-anchor", "end");
    nodes.gate2Label.setAttribute("y", geometry.padTop + 16);
    nodes.gate2Label.textContent = `frontier judge · ${money(PRICES.deep * 1000, 2)}/1k`;

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

    // Zone labels sit centred between the gates, well clear of the drag handles.
    const zoneX = (gate1 + gate2) / 2;
    for (const node of [nodes.zoneBlock, nodes.zoneClear, nodes.zoneEsc]) {
      node.style.display = cascade ? "" : "none";
      node.setAttribute("x", zoneX);
      node.setAttribute("text-anchor", "middle");
    }
    nodes.zoneBlock.setAttribute("y", Math.max(geometry.padTop + 16, yBlock - 12));
    nodes.zoneBlock.textContent = "BLOCKED AT THE SCREEN";
    nodes.zoneEsc.setAttribute("y", (yWarn + yBlock) / 2 + 3);
    nodes.zoneEsc.textContent = "UNCERTAIN → PAYS THE JUDGE";
    nodes.zoneClear.setAttribute("y", Math.min(H - geometry.padBottom - 8, yWarn + 20));
    nodes.zoneClear.textContent = "CLEARED — NEVER PAYS";
  }

  function attachDrag(group, which) {
    let dragging = false;
    group.addEventListener("pointerdown", (event) => {
      dragging = true;
      group.setPointerCapture(event.pointerId);
      event.preventDefault();
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
    decisions.length = 0; // stale shares describe a policy that no longer exists
    layoutOverlay();
    if (reducedMotion) renderStatic();
  }

  // ── meters ──────────────────────────────────────────────
  const meterFlat = document.getElementById("meter-flat");
  const meterCascade = document.getElementById("meter-cascade");
  const meterEsc = document.getElementById("meter-esc");
  const meterSave = document.getElementById("meter-save");
  meterFlat.textContent = money(PRICES.deep * 1000, 2);

  function updateMeters() {
    if (mode === "flat" || decisions.length < 60) {
      if (mode === "flat") {
        meterCascade.textContent = "—";
        meterEsc.textContent = "—";
        meterSave.textContent = "—";
      }
      return;
    }
    let escalated = 0;
    for (const d of decisions) if (d === 1) escalated += 1;
    const share = escalated / decisions.length;
    const costCascade = 1000 * (PRICES.cheap + share * PRICES.deep);
    const costFlat = 1000 * PRICES.deep;
    meterCascade.textContent = money(costCascade, 2);
    meterEsc.textContent = (share * 100).toFixed(1) + "%";
    meterSave.textContent = "−" + ((1 - costCascade / costFlat) * 100).toFixed(1) + "%";
  }

  // ── controls ────────────────────────────────────────────
  const modeFlat = document.getElementById("mode-flat");
  const modeCascade = document.getElementById("mode-cascade");
  function setMode(next) {
    mode = next;
    modeFlat.setAttribute("aria-pressed", String(mode === "flat"));
    modeCascade.setAttribute("aria-pressed", String(mode === "cascade"));
    decisions.length = 0;
    for (const p of particles) if (p.state !== BURSTING) {
      // Re-route everything left of the decided gates under the new regime.
      if (p.x < (mode === "cascade" ? geometry.gate1() : geometry.gate2())) p.state = FLOW;
    }
    layoutOverlay();
    updateMeters();
    if (reducedMotion) renderStatic();
  }
  modeFlat.addEventListener("click", () => setMode("flat"));
  modeCascade.addEventListener("click", () => setMode("cascade"));

  function applyPreset(name) {
    traffic = TRAFFIC[name];
    [warn, block] = traffic.band;
    decisions.length = 0;
    for (const p of particles) {
      if (p.x < geometry.gate1() || p.state === BURSTING) {
        const s = sampleScores();
        p.unsafe = s.unsafe; p.cheap = s.cheap; p.deep = s.deep;
        if (p.state !== BURSTING) p.state = FLOW;
      }
    }
    if (mode !== "cascade") setMode("cascade");
    layoutOverlay();
    if (reducedMotion) renderStatic();
  }
  document.getElementById("preset-tight").addEventListener("click", () => applyPreset("easy"));
  document.getElementById("preset-wide").addEventListener("click", () => applyPreset("adversarial"));

  // ── lifecycle ───────────────────────────────────────────
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const hero = canvas.parentElement;
    W = hero.clientWidth;
    H = hero.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    overlay.setAttribute("width", W);
    overlay.setAttribute("height", H);
    overlay.setAttribute("viewBox", `0 0 ${W} ${H}`);
    layoutOverlay();
  }

  function renderStatic() {
    // Reduced motion: simulate ~14 seconds silently, then paint one honest frame.
    decisions.length = 0;
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
  window.addEventListener("resize", resize);

  if (reducedMotion) {
    renderStatic();
    return;
  }

  let last = performance.now();
  let meterClock = 0;
  let running = true;
  // Don't burn the battery when the hero is off screen.
  new IntersectionObserver((entries) => { running = entries[0].isIntersecting; })
    .observe(canvas);

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

function buildProgram(gl) {
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
  return program;
}
