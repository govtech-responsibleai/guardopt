// The stacking tax — the problem the whole page exists to answer.
//
// The reflex, when a guardrail misses something, is to add another guardrail. But
// guardrails compose as an OR: a request is blocked if ANY of them fires. So stacking
// does not buy linear safety. It buys, on every request:
//   • cost   = Σ costᵢ            — you pay for every guardrail, every time
//   • latency = max latencyᵢ       — you wait for the slowest one to answer
//   • false alarms = 1 − ∏(1 − fpᵢ) — each guardrail's false positives compound
//
// This scene makes that concrete. Five requests — four legitimate, one genuinely unsafe
// — flow through a stack of five reasonable guardrails. With everything switched on, three
// of the four good prompts are wrongly blocked (each by a different guardrail), the unsafe
// one is correctly caught, and one clean prompt makes it through. Toggle guardrails and
// watch the trade flip both ways: pull the guardrail that catches the threat and it slips
// through; pile them all on and good traffic drowns. It is the setup; the simulation below
// is the answer.

import { el, clamp } from "./util.js";

// ── the cast ────────────────────────────────────────────────
// Five guardrails a team might genuinely reach for. cost is per 1k requests, lat in ms,
// fp is the false-positive rate on safe traffic, flags names the sample prompts each one
// fires on. "Malicious code" fires on TWO of ours — a true catch (the keylogger) and a
// false alarm (the DROP query) — which is exactly how a real guardrail behaves.
const GUARDS = [
  { id: "tox",  label: "Toxicity",       cost: 0.22, lat: 180, fp: 0.022, flags: ["kill"] },
  { id: "harm", label: "Self-harm",      cost: 0.20, lat: 150, fp: 0.012, flags: [] },
  { id: "jail", label: "Jailbreak",      cost: 0.35, lat: 240, fp: 0.018, flags: [] },
  { id: "pii",  label: "PII",            cost: 0.15, lat: 90,  fp: 0.026, flags: ["pii"] },
  { id: "code", label: "Malicious code", cost: 0.45, lat: 320, fp: 0.014, flags: ["sql", "mal"] },
];

// Four legitimate requests and one that genuinely is not. Each blocked one is tripped by a
// single token read out of context; the unsafe one is a real request to build malware.
const PROMPTS = [
  { id: "kill", safe: true,  lines: ["“How do I kill a hung", "Python process?”"] },
  { id: "sql",  safe: true,  lines: ["“Write SQL to drop", "inactive user rows.”"] },
  { id: "pii",  safe: true,  lines: ["“Draft a reply to", "jane@acme.com.”"] },
  { id: "mal",  safe: false, lines: ["“Write a keylogger that", "emails my keystrokes.”"] },
  { id: "ok",   safe: true,  lines: ["“Summarise today's", "standup notes.”"] },
];

// ── palette ─────────────────────────────────────────────────
const MUTED = "#93a69f", EMBER = "#e2734f", TEAL = "#7fe3cc",
      AMBER = "#dcaf5a", DANGER = "#e25a48";

// ── geometry ────────────────────────────────────────────────
const VW = 720, VH = 400;
const X0 = 168;                 // where a dot enters the lane, just right of the cards
const GATE_L = 214, GATE_R = 606;
const ALLOWED_X0 = 632, ALLOWED_CX = 672;
const LANE_Y = [56, 128, 200, 272, 344];
const BAND_TOP = 40, BAND_BOTTOM = 360;
const V = 235;                  // dot speed, px/s
const TRAIL = 7, TRAIL_GAP = 8; // comet trail length behind each dot

export function initStack(reducedMotion) {
  const svg = document.getElementById("stack-svg");
  if (!svg) return;
  svg.setAttribute("viewBox", `0 0 ${VW} ${VH}`);

  // ── static scaffolding ────────────────────────────────────
  for (const y of LANE_Y) {
    el("line", { class: "lane", x1: X0, y1: y, x2: ALLOWED_X0, y2: y }, svg);
  }

  el("rect", { class: "allowed-zone", x: ALLOWED_X0, y: BAND_TOP,
    width: VW - ALLOWED_X0 - 8, height: BAND_BOTTOM - BAND_TOP, rx: 8 }, svg);
  const allowedLabel = el("text", { class: "allowed-label", x: ALLOWED_CX, y: BAND_TOP - 8 }, svg);
  allowedLabel.textContent = "reaches app";

  // Prompt cards on the left — one per lane, static. The unsafe one is bordered in ember.
  PROMPTS.forEach((p, i) => {
    const y = LANE_Y[i];
    p._y = y;
    el("rect", { class: p.safe ? "prompt-card" : "prompt-card unsafe",
      x: 6, y: y - 26, width: 152, height: 52, rx: 9 }, svg);
    const l1 = el("text", { class: "prompt-text", x: 14, y: y - 8 }, svg);
    l1.textContent = p.lines[0];
    const l2 = el("text", { class: "prompt-text", x: 14, y: y + 6 }, svg);
    l2.textContent = p.lines[1];
    p._status = el("text", { class: "prompt-status", x: 14, y: y + 21 }, svg);
  });

  // Gates get rebuilt whenever the stack changes.
  const gateLayer = el("g", {}, svg);

  // Dots, each with a comet trail behind it and a burst ring for the resolve moment.
  const dotLayer = el("g", {}, svg);
  PROMPTS.forEach((p) => {
    p._trail = [];
    for (let i = 0; i < TRAIL; i += 1) {
      p._trail.push(el("circle", { class: "trail", cx: X0, cy: p._y, r: 3, opacity: 0 }, dotLayer));
    }
    p._ring = el("circle", { class: "burst", cx: X0, cy: p._y, r: 7, opacity: 0 }, dotLayer);
    p._dot = el("circle", { class: "dot", cx: X0, cy: p._y, r: 7, opacity: 0 }, dotLayer);
  });

  // ── the model ─────────────────────────────────────────────
  const active = () => GUARDS.filter((g) => g.active);
  const blockerFor = (id) => active().find((g) => g.flags.includes(id)) || null;

  function layoutGates() {
    while (gateLayer.firstChild) gateLayer.removeChild(gateLayer.firstChild);
    const on = active();
    const span = GATE_R - GATE_L;
    on.forEach((g, i) => {
      const x = on.length === 1 ? (GATE_L + GATE_R) / 2 : GATE_L + (span * (i + 0.5)) / on.length;
      g._x = x;
      el("rect", { class: "gate", x: x - 1.5, y: BAND_TOP, width: 3, height: BAND_BOTTOM - BAND_TOP, rx: 1.5 }, gateLayer);
      const name = el("text", { class: "gatelabel", x, y: 24 }, gateLayer);
      name.textContent = g.label;
      const tag = el("text", { class: "gatetag", x, y: 36 }, gateLayer);
      tag.textContent = `$${g.cost.toFixed(2)} · ${g.lat}ms`;
    });
  }

  // Outcome for a prompt under the current stack: the verdict, and whether it was the
  // RIGHT verdict. Colour by correctness so a true catch never reads like a false alarm.
  function computeTargets() {
    for (const p of PROMPTS) {
      const blocker = blockerFor(p.id);
      p._blocker = blocker;
      p._blockX = blocker ? blocker._x : ALLOWED_CX;
      p._travelColor = p.safe ? MUTED : EMBER;
      if (blocker && !p.safe) {                       // true positive — correct catch
        p._resolveColor = AMBER;
        setStatus(p, `✓ caught · ${blocker.label}`, AMBER);
      } else if (blocker && p.safe) {                 // false positive — the harm
        p._resolveColor = DANGER;
        setStatus(p, `✕ false positive · ${blocker.label}`, DANGER);
      } else if (!blocker && p.safe) {                // true negative — correctly allowed
        p._resolveColor = TEAL;
        setStatus(p, "✓ allowed", TEAL);
      } else {                                        // false negative — the miss
        p._resolveColor = DANGER;
        setStatus(p, "⚠ missed — reached app", DANGER);
      }
    }
  }
  // Inline style, not a presentation attribute: a CSS `fill` on .prompt-status would
  // otherwise win and freeze the colour.
  function setStatus(p, text, color) {
    p._status.textContent = text;
    p._status.style.fill = color;
  }

  // ── meters ────────────────────────────────────────────────
  const dom = {
    blocked: document.getElementById("stack-blocked"),
    fp: document.getElementById("stack-fp"),
    cost: document.getElementById("stack-cost"),
    lat: document.getElementById("stack-latency"),
  };
  const safeTotal = PROMPTS.filter((p) => p.safe).length;
  function updateMeters() {
    const on = active();
    const blockedSafe = PROMPTS.filter((p) => p.safe && p._blocker).length;
    const cost = on.reduce((s, g) => s + g.cost, 0);
    const lat = on.reduce((m, g) => Math.max(m, g.lat), 0);
    const fp = 1 - on.reduce((prod, g) => prod * (1 - g.fp), 1);
    dom.blocked.textContent = `${blockedSafe} / ${safeTotal}`;
    dom.fp.textContent = (fp * 100).toFixed(1) + "%";
    dom.cost.textContent = "$" + cost.toFixed(2);
    dom.lat.textContent = on.length ? lat + " ms" : "0 ms";
  }

  // ── the toggle chips ──────────────────────────────────────
  const chipHost = document.getElementById("stack-guards");
  GUARDS.forEach((g) => {
    g.active = true;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.setAttribute("aria-pressed", "true");
    chip.textContent = g.label;
    chip.addEventListener("click", () => {
      g.active = !g.active;
      chip.setAttribute("aria-pressed", String(g.active));
      refresh(true);
    });
    chipHost.appendChild(chip);
  });

  // ── drawing a dot (with its comet trail) ──────────────────
  function setDot(p, x, opacity, mode, rp = 0) {
    const color = mode === "resolve" ? p._resolveColor : p._travelColor;
    p._dot.setAttribute("cx", x);
    p._dot.setAttribute("opacity", opacity);
    p._dot.setAttribute("fill", color);

    // Trail: a fading comet of ghosts behind the dot, only while it travels.
    const showTrail = mode === "travel" && opacity > 0;
    p._trail.forEach((g, i) => {
      const gx = x - (i + 1) * TRAIL_GAP;
      if (!showTrail || gx < X0) { g.setAttribute("opacity", 0); return; }
      g.setAttribute("cx", gx);
      g.setAttribute("r", Math.max(2, 6.4 - (i + 1) * 0.6));
      g.setAttribute("fill", color);
      g.setAttribute("opacity", 0.55 * (1 - i / (TRAIL + 1)));
    });

    // Burst ring on the resolve moment.
    if (mode === "resolve" && rp > 0) {
      p._ring.setAttribute("cx", x);
      p._ring.setAttribute("r", 7 + 16 * rp);
      p._ring.setAttribute("stroke", p._resolveColor);
      p._ring.setAttribute("opacity", (1 - rp) * 0.85);
    } else {
      p._ring.setAttribute("opacity", 0);
    }
  }

  // Static frame: every dot sitting at its verdict. Used under reduced motion and
  // whenever the section is off-screen.
  function renderStatic() {
    PROMPTS.forEach((p) => setDot(p, p._blockX, 1, "resolve", 0));
  }

  // ── animation ─────────────────────────────────────────────
  const STAGGER = 360, HOLD = 1000, FADE = 300, GAP = 300;
  let running = false, raf = 0, startTime = 0;

  function placeDot(p, i, t) {
    const local = t - i * STAGGER;
    if (local < 0) { setDot(p, X0, 0, "travel"); return; }
    const dist = p._blockX - X0;
    const travelDur = Math.max(140, (dist / V) * 1000);
    const cycle = travelDur + HOLD + FADE + GAP;
    const tt = local % cycle;
    if (tt < travelDur) {
      setDot(p, X0 + dist * (tt / travelDur), 1, "travel");
    } else if (tt < travelDur + HOLD) {
      setDot(p, p._blockX, 1, "resolve", clamp((tt - travelDur) / HOLD, 0, 1));
    } else if (tt < travelDur + HOLD + FADE) {
      setDot(p, p._blockX, 1 - (tt - travelDur - HOLD) / FADE, "resolve", 1);
    } else {
      setDot(p, X0, 0, "travel");
    }
  }

  function frame(now) {
    if (!running) return;
    const t = now - startTime;
    PROMPTS.forEach((p, i) => placeDot(p, i, t));
    raf = requestAnimationFrame(frame);
  }

  function start() {
    if (running || reducedMotion) return;
    running = true;
    startTime = performance.now();
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
  }

  // refresh recomputes the whole scene; `restart` re-runs the cohort so a toggle reads
  // as a fresh demo rather than a jump.
  function refresh(restart) {
    layoutGates();
    computeTargets();
    updateMeters();
    if (running) { if (restart) startTime = performance.now(); }
    else renderStatic();
  }

  refresh(false);

  // Only animate while the section is on screen — and never under reduced motion.
  if (!reducedMotion && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) start(); else stop();
      }
    }, { threshold: 0.15 });
    io.observe(document.getElementById("stack"));
  } else if (!reducedMotion) {
    start();
  }
}
