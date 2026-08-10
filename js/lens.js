// A3 — the score-space lens. Every labelled request as a dot in two guardrails'
// score space; the policy as draggable lines; the block region as geometry.
//
// The point this scene exists to make: a flat OR-policy can only carve rectilinear
// unions, so the "theatrical villainy" cluster (screen says risky, judge says fine)
// is unreachable without collateral damage. The cascade's band routes exactly that
// corridor to the judge — a region shape no flat policy can express — and the search
// buttons find both optima live, by brute force, in front of you.

import { mulberry32, gaussian, clamp, tween, PRICES, el } from "./util.js";

const VB_W = 640, VB_H = 560;
const M = { left: 50, right: 18, top: 18, bottom: 46 };
const PLOT_W = VB_W - M.left - M.right;
const PLOT_H = VB_H - M.top - M.bottom;

const toX = (s) => M.left + s * PLOT_W;
const toY = (s) => M.top + (1 - s) * PLOT_H;
const fromX = (x) => clamp((x - M.left) / PLOT_W, 0, 1);
const fromY = (y) => clamp(1 - (y - M.top) / PLOT_H, 0, 1);

function buildCases() {
  const rng = mulberry32(4471);
  const cases = [];
  const add = (n, unsafe, cheapMean, cheapSd, deepMean, deepSd) => {
    for (let i = 0; i < n; i += 1) {
      cases.push({
        unsafe,
        cheap: clamp(cheapMean + gaussian(rng) * cheapSd, 0.02, 0.98),
        deep: clamp(deepMean + gaussian(rng) * deepSd, 0.02, 0.98),
      });
    }
  };
  add(140, false, 0.3, 0.13, 0.24, 0.1);   // ordinary safe traffic
  add(70, true, 0.72, 0.13, 0.79, 0.09);   // ordinary unsafe traffic
  add(18, false, 0.68, 0.07, 0.2, 0.07);   // theatrical villainy: screen risky, judge fine
  add(12, true, 0.3, 0.08, 0.82, 0.06);    // screen-blind unsafe: only the judge sees it
  return cases;
}

export function initLens(reducedMotion) {
  const svg = document.getElementById("lens-svg");
  const cases = buildCases();

  // Logical policy state; `visual` trails it during morph animations.
  const state = { mode: "flat", t1: 0.55, t2: 0.55, warn: 0.42, block: 0.66 };
  const visual = { ...state };

  // ── static scaffolding ──────────────────────────────────
  const regionLayer = el("g", {}, svg);
  const rBlockRight = el("rect", { class: "region-block" }, regionLayer);
  const rBlockLeft = el("rect", { class: "region-block" }, regionLayer);
  const rCorridor = el("rect", { class: "region-corridor" }, regionLayer);

  el("line", { x1: M.left, y1: toY(0), x2: toX(1), y2: toY(0), stroke: "#243029" }, svg);
  el("line", { x1: M.left, y1: M.top, x2: M.left, y2: toY(0), stroke: "#243029" }, svg);
  const xAxis = el("text", { class: "axis-label", x: M.left + PLOT_W / 2, y: VB_H - 12, "text-anchor": "middle" }, svg);
  xAxis.textContent = `cheap screen score → (${(PRICES.cheap * 1000).toFixed(3)}$/1k)`;
  const yAxis = el("text", { class: "axis-label", x: 14, y: M.top + PLOT_H / 2, "text-anchor": "middle",
    transform: `rotate(-90 14 ${M.top + PLOT_H / 2})` }, svg);
  yAxis.textContent = `expensive judge score → (${(PRICES.deep * 1000).toFixed(2)}$/1k)`;

  const dotLayer = el("g", {}, svg);
  const dots = cases.map((c) => el("circle", { class: "case", cx: toX(c.cheap), cy: toY(c.deep), r: 4.2 }, dotLayer));

  const lineLayer = el("g", {}, svg);
  const lines = {
    t1: makeDraggable("t1", "vertical", "thr", "block ≥"),
    warn: makeDraggable("warn", "vertical", "thr warn", "escalate ≥"),
    block: makeDraggable("block", "vertical", "thr", "block ≥"),
    t2: makeDraggable("t2", "horizontal", "thr", "judge blocks ≥"),
  };

  function makeDraggable(key, orientation, cls, labelPrefix) {
    const group = el("g", { tabindex: 0, role: "slider", "aria-orientation": orientation,
      "aria-label": `${labelPrefix} threshold` }, lineLayer);
    const hit = el("line", { class: "thr-hit" }, group);
    const line = el("line", { class: cls }, group);
    const label = el("text", { class: cls.includes("warn") ? "thr-label warn" : "thr-label" }, group);

    let dragging = false;
    group.addEventListener("pointerdown", (event) => {
      dragging = true; group.setPointerCapture(event.pointerId); event.preventDefault();
    });
    group.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const rect = svg.getBoundingClientRect();
      const value = orientation === "vertical"
        ? fromX((event.clientX - rect.left) * (VB_W / rect.width))
        : fromY((event.clientY - rect.top) * (VB_H / rect.height));
      setValue(key, value);
    });
    group.addEventListener("pointerup", () => { dragging = false; });
    group.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 0.02, ArrowLeft: -0.02, ArrowUp: 0.02, ArrowDown: -0.02 }[event.key];
      if (!step) return;
      event.preventDefault();
      setValue(key, state[key] + step);
    });
    return { group, hit, line, label, labelPrefix, orientation };
  }

  function setValue(key, value) {
    if (key === "warn") state.warn = clamp(value, 0.03, state.block - 0.04);
    else if (key === "block") state.block = clamp(value, state.warn + 0.04, 0.97);
    else state[key] = clamp(value, 0.03, 0.97);
    Object.assign(visual, state);
    render();
  }

  // ── verdicts & metrics ──────────────────────────────────
  function verdict(c, s) {
    if (s.mode === "flat") return c.cheap >= s.t1 || c.deep >= s.t2;
    if (c.cheap >= s.block) return true;
    if (c.cheap < s.warn) return false;
    return c.deep >= s.t2;
  }
  const consulted = (c, s) => s.mode === "flat" || (c.cheap >= s.warn && c.cheap < s.block);

  function metrics(s) {
    let tp = 0, fp = 0, fn = 0, consult = 0;
    for (const c of cases) {
      const blocked = verdict(c, s);
      if (blocked && c.unsafe) tp += 1;
      else if (blocked) fp += 1;
      else if (c.unsafe) fn += 1;
      if (consulted(c, s)) consult += 1;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall = tp + fn > 0 ? tp / (tp + fn) : null;
    const f1 = precision === null || recall === null || precision + recall === 0
      ? null : (2 * precision * recall) / (precision + recall);
    const share = consult / cases.length;
    const costPer1k = 1000 * (PRICES.cheap + share * PRICES.deep);
    return { precision, recall, f1, share, costPer1k };
  }

  const dom = {
    precision: document.getElementById("lens-precision"),
    recall: document.getElementById("lens-recall"),
    f1: document.getElementById("lens-f1"),
    cost: document.getElementById("lens-cost"),
    consulted: document.getElementById("lens-consulted"),
    note: document.getElementById("lens-note"),
    modeFlat: document.getElementById("lens-flat"),
    modeCascade: document.getElementById("lens-cascade"),
  };
  const pct = (v) => (v === null ? "—" : (v * 100).toFixed(1) + "%");

  // ── render ──────────────────────────────────────────────
  function render() {
    const s = visual;
    const cascade = s.mode === "cascade";

    if (cascade) {
      const xWarn = toX(s.warn), xBlock = toX(s.block), yT2 = toY(s.t2);
      rBlockRight.setAttribute("x", xBlock); rBlockRight.setAttribute("y", M.top);
      rBlockRight.setAttribute("width", Math.max(0, toX(1) - xBlock));
      rBlockRight.setAttribute("height", PLOT_H);
      rBlockLeft.setAttribute("x", xWarn); rBlockLeft.setAttribute("y", M.top);
      rBlockLeft.setAttribute("width", Math.max(0, xBlock - xWarn));
      rBlockLeft.setAttribute("height", Math.max(0, yT2 - M.top));
      rCorridor.style.display = "";
      rCorridor.setAttribute("x", xWarn); rCorridor.setAttribute("y", M.top);
      rCorridor.setAttribute("width", Math.max(0, xBlock - xWarn));
      rCorridor.setAttribute("height", PLOT_H);
    } else {
      const xT1 = toX(s.t1), yT2 = toY(s.t2);
      rBlockRight.setAttribute("x", xT1); rBlockRight.setAttribute("y", M.top);
      rBlockRight.setAttribute("width", Math.max(0, toX(1) - xT1));
      rBlockRight.setAttribute("height", PLOT_H);
      rBlockLeft.setAttribute("x", M.left); rBlockLeft.setAttribute("y", M.top);
      rBlockLeft.setAttribute("width", Math.max(0, xT1 - M.left));
      rBlockLeft.setAttribute("height", Math.max(0, yT2 - M.top));
      rCorridor.style.display = "none";
    }

    placeLine(lines.t1, toX(s.t1), "vertical", !cascade, s.t1);
    placeLine(lines.warn, toX(s.warn), "vertical", cascade, s.warn);
    placeLine(lines.block, toX(s.block), "vertical", cascade, s.block);
    placeLine(lines.t2, toY(s.t2), "horizontal", true, s.t2, cascade);

    // Dots recolour by outcome under the LOGICAL policy (not the tweened one):
    // false positives and misses are the loud ones.
    for (let i = 0; i < cases.length; i += 1) {
      const c = cases[i];
      const blocked = verdict(c, state);
      const dot = dots[i];
      if (blocked && c.unsafe) { dot.setAttribute("fill", "#a34d33"); dot.setAttribute("stroke", "none"); }
      else if (!blocked && !c.unsafe) { dot.setAttribute("fill", "#3e6f62"); dot.setAttribute("stroke", "none"); }
      else if (blocked) { dot.setAttribute("fill", "#2a201c"); dot.setAttribute("stroke", "#e2734f"); }
      else { dot.setAttribute("fill", "#241412"); dot.setAttribute("stroke", "#e25a48"); }
    }

    const m = metrics(state);
    dom.precision.textContent = pct(m.precision);
    dom.recall.textContent = pct(m.recall);
    dom.f1.textContent = m.f1 === null ? "—" : m.f1.toFixed(3);
    dom.cost.textContent = "$" + m.costPer1k.toFixed(2) + " /1k";
    dom.consulted.textContent = pct(m.share) + " of requests";
  }

  function placeLine(entry, position, orientation, show, value, corridorOnly = false) {
    entry.group.style.display = show ? "" : "none";
    if (!show) return;
    const attrs = orientation === "vertical"
      ? { x1: position, x2: position, y1: M.top, y2: toY(0) }
      : {
          x1: corridorOnly ? toX(visual.warn) : M.left,
          x2: corridorOnly ? toX(visual.block) : toX(1),
          y1: position, y2: position,
        };
    for (const node of [entry.hit, entry.line]) {
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    }
    entry.group.setAttribute("aria-valuenow", value.toFixed(2));
    if (orientation === "vertical") {
      entry.label.setAttribute("x", position + 4);
      entry.label.setAttribute("y", M.top + 12);
      entry.label.textContent = `${entry.labelPrefix} ${value.toFixed(2)}`;
    } else {
      entry.label.setAttribute("x", (corridorOnly ? toX(visual.warn) : M.left) + 4);
      entry.label.setAttribute("y", position - 5);
      entry.label.textContent = `${entry.labelPrefix} ${value.toFixed(2)}`;
    }
  }

  // ── mode switching, with the morph ──────────────────────
  function setMode(mode) {
    if (state.mode === mode) return;
    const from = { ...state };
    state.mode = mode;
    if (mode === "cascade") {
      state.warn = clamp(from.t1 - 0.14, 0.03, 0.9);
      state.block = clamp(from.t1 + 0.06, state.warn + 0.04, 0.97);
    } else {
      state.t1 = clamp((from.warn + from.block) / 2, 0.03, 0.97);
    }
    dom.modeFlat.setAttribute("aria-pressed", String(mode === "flat"));
    dom.modeCascade.setAttribute("aria-pressed", String(mode === "cascade"));
    dom.note.textContent = mode === "cascade"
      ? "Dots left of the amber line never consult the judge at all — that column of "
        + "the bill simply disappears. And notice the lower-right cluster: risky to the "
        + "screen, fine to the judge. Only the corridor can clear them without blocking."
      : "A flat policy ORs the guardrails: block right of the teal line or above it. "
        + "Both guardrails run on every request, so the bill is fixed — and the "
        + "lower-right cluster is unreachable without collateral blocks.";
    morphTo(state);
  }
  dom.modeFlat.addEventListener("click", () => setMode("flat"));
  dom.modeCascade.addEventListener("click", () => setMode("cascade"));

  function morphTo(target) {
    if (reducedMotion) { Object.assign(visual, target); render(); return; }
    const from = { ...visual };
    visual.mode = target.mode; // regions swap shape immediately; positions tween
    tween({
      duration: 480,
      onUpdate: (t) => {
        for (const key of ["t1", "t2", "warn", "block"]) {
          visual[key] = from[key] + (target[key] - from[key]) * t;
        }
        render();
      },
      onDone: () => { Object.assign(visual, target); render(); },
    });
  }

  // ── the live searches ───────────────────────────────────
  const GRID = 41;
  const gridValue = (i) => i / (GRID - 1);

  document.getElementById("lens-best-flat").addEventListener("click", () => {
    let best = null;
    for (let i = 0; i < GRID; i += 1) {
      for (let j = 0; j < GRID; j += 1) {
        const candidate = { mode: "flat", t1: gridValue(i), t2: gridValue(j), warn: state.warn, block: state.block };
        const m = metrics(candidate);
        if (m.f1 !== null && (best === null || m.f1 > best.m.f1 + 1e-9)) best = { candidate, m };
      }
    }
    if (!best) return;
    state.mode = "flat";
    dom.modeFlat.setAttribute("aria-pressed", "true");
    dom.modeCascade.setAttribute("aria-pressed", "false");
    Object.assign(state, { t1: best.candidate.t1, t2: best.candidate.t2 });
    morphTo({ ...state });
  });

  document.getElementById("lens-best-cascade").addEventListener("click", () => {
    let best = null;
    for (let w = 0; w < GRID; w += 1) {
      for (let b = w + 2; b < GRID; b += 1) {
        for (let j = 0; j < GRID; j += 1) {
          const candidate = { mode: "cascade", warn: gridValue(w), block: gridValue(b), t2: gridValue(j), t1: state.t1 };
          const m = metrics(candidate);
          if (m.f1 === null) continue;
          // Best F1; ties go to the cheaper corridor — the optimiser's own manner.
          if (best === null || m.f1 > best.m.f1 + 1e-9
            || (Math.abs(m.f1 - best.m.f1) <= 1e-9 && m.costPer1k < best.m.costPer1k)) {
            best = { candidate, m };
          }
        }
      }
    }
    if (!best) return;
    state.mode = "cascade";
    dom.modeFlat.setAttribute("aria-pressed", "false");
    dom.modeCascade.setAttribute("aria-pressed", "true");
    Object.assign(state, { warn: best.candidate.warn, block: best.candidate.block, t2: best.candidate.t2 });
    morphTo({ ...state });
  });

  Object.assign(visual, state);
  render();
}
