// The search — how guardopt actually finds the cascade, told as the five steps it runs.
//
// This replaces the old "explore the trade" scenes. It is not a toy: every step is a real
// stage of the optimiser, in order.
//
//   1. Score      — start from labelled traffic you already scored. Nothing runs here.
//   2. Candidates — thresholds are read OUT of the data: a candidate sits at every point
//                   the label flips (the midpoint between an adjacent safe and unsafe
//                   score is the decision boundary). Never a blind grid.
//   3. Replay     — route every case through one cascade and count the confusion matrix;
//                   the escalate band is the only traffic the judge ever sees.
//   4. Frontier   — score every legal cascade the same way, drop the dominated, keep the
//                   Pareto frontier (nothing beats a frontier policy on both axes).
//   5. Pick       — three points on it: Minimal (F0.5), Balanced (F1), Strict (F2), each
//                   shipped with honest error bars.
//
// The dataset is synthetic — your traffic is the thing worth measuring — but the machinery
// is the real one: candidates from label transitions, exhaustive scoring, Pareto keep,
// F-score picks.

import { mulberry32, gaussian, clamp, tween, easeOutCubic, PRICES, el } from "./util.js";

// ── palette ─────────────────────────────────────────────────
const SAFE = "#4f9c88", UNSAFE = "#cf6f4a";
const ALLOW_OK = "#3e6f62", BLOCK_OK = "#a34d33";
const TEAL = "#5bc4ad", TEAL_BRIGHT = "#7fe3cc", AMBER = "#dcaf5a",
      EMBER = "#e2734f", DANGER = "#e25a48", GREY = "#39443e", FAINT = "#5e6f69";

// ── geometry ────────────────────────────────────────────────
const VBW = 680, VBH = 430;
const M = { left: 46, right: 18, top: 24, bottom: 40 };
const PW = VBW - M.left - M.right, PH = VBH - M.top - M.bottom;
const toX = (s) => M.left + s * PW;
const toY = (s) => M.top + (1 - s) * PH;

const STEPS = [
  { title: "Score your labelled traffic",
    cap: "Start from traffic you've already scored and labelled. Each dot is one request — right means the cheap screen scored it riskier, up means the expensive judge did. guardopt runs no guardrails here; it works entirely from scores that already exist." },
  { title: "Draw candidates where labels flip",
    cap: "Thresholds are read out of your data, not swept on a grid. guardopt puts a candidate exactly at each point the label changes — the midpoint between an adjacent safe and unsafe score is the decision boundary, and the single most informative place to cut. The teal lines are screen candidates; the amber line is the judge's." },
  { title: "Replay & score one cascade",
    cap: "Take one cascade — block right of the solid teal line, allow left of the amber line, and escalate only the band between them to the judge, who decides on the amber line. Route every labelled case through it and count. The allow region never consults the judge, so that whole column of the bill disappears." },
  { title: "Enumerate & keep the frontier",
    cap: "Now score every legal cascade the same way — the space is sized first, then enumerated exactly (or refused with the number, never a silent sample). Each dot is a policy someone could ship, placed by its precision and recall. Drop every one another policy beats on both; what remains is the Pareto frontier — where you cannot raise precision without losing recall." },
  { title: "Pick Minimal · Balanced · Strict",
    cap: "guardopt does not pick one — which policy is best depends on what a miss costs you. It returns three points on the frontier: Minimal (precision-weighted F0.5), Balanced (F1), Strict (recall-weighted F2) — each with Wilson intervals, a holdout check, and bootstrap stability attached." },
];

export function initSearch(reducedMotion) {
  const svg = document.getElementById("search-svg");
  if (!svg) return;
  svg.setAttribute("viewBox", `0 0 ${VBW} ${VBH}`);

  // ── the data (synthetic, deterministic) ───────────────────
  const CASES = buildCases();

  // Chosen cascade for step 3 + the whole frontier + the three picks.
  const { policies, frontier, picks } = search();
  const chosen = picks.balanced || frontier[0];

  // ── scaffolding: axes ─────────────────────────────────────
  el("line", { x1: M.left, y1: toY(0), x2: toX(1), y2: toY(0), stroke: "#243029" }, svg);
  el("line", { x1: M.left, y1: M.top, x2: M.left, y2: toY(0), stroke: "#243029" }, svg);
  const xLabel = el("text", { class: "s-axis", x: M.left + PW / 2, y: VBH - 10, "text-anchor": "middle" }, svg);
  const yLabel = el("text", { class: "s-axis", x: 13, y: M.top + PH / 2, "text-anchor": "middle",
    transform: `rotate(-90 13 ${M.top + PH / 2})` }, svg);

  // ── scatter view (steps 1–3) ──────────────────────────────
  const scatterG = el("g", {}, svg);
  const regionG = el("g", {}, scatterG);
  const rAllow = el("rect", { class: "s-allow" }, regionG);
  const rBand = el("rect", { class: "s-band" }, regionG);
  const rBlock = el("rect", { class: "s-block" }, regionG);
  const candG = el("g", {}, scatterG);
  const lineG = el("g", {}, scatterG);
  const lWarn = el("line", { class: "s-thr amber" }, lineG);
  const lBlock = el("line", { class: "s-thr" }, lineG);
  const lJudge = el("line", { class: "s-thr amber judge" }, lineG);
  const dotG = el("g", {}, scatterG);
  const dots = CASES.map((c) => el("circle", { class: "s-dot", cx: toX(c.cheap), cy: toY(c.judge), r: 4 }, dotG));

  // ── frontier view (steps 4–5) ─────────────────────────────
  const frontierG = el("g", { opacity: 0 }, svg);
  layoutFrontier();
  const polDots = policies.map((p) =>
    el("circle", { class: "f-dot", cx: toX(p._x), cy: toY(p._y), r: 3.4 }, frontierG));
  const ringG = el("g", {}, frontierG);

  // ── side panel ────────────────────────────────────────────
  const readout = document.getElementById("search-readout");
  const railItems = Array.from(document.querySelectorAll("#search-steps .s-step"));
  const caption = document.getElementById("search-caption");

  let step = 0, timer = 0;

  // ── per-step rendering ────────────────────────────────────
  function renderStep(n, animate) {
    step = n;
    railItems.forEach((it, i) => it.classList.toggle("active", i === n));
    caption.textContent = STEPS[n].cap;

    const onScatter = n <= 2;
    setGroupOpacity(scatterG, onScatter ? 1 : 0, animate);
    setGroupOpacity(frontierG, onScatter ? 0 : 1, animate);
    xLabel.textContent = onScatter ? "cheap screen score →" : "precision →";
    yLabel.textContent = onScatter ? "expensive judge score →" : "recall →";

    // candidates (step ≥ 1)
    candG.style.display = n === 1 ? "" : "none";

    // regions + threshold lines + outcome recolour (step 3)
    const showCascade = n === 2;
    regionG.style.display = showCascade ? "" : "none";
    lineG.style.display = showCascade ? "" : "none";
    if (showCascade) drawCascade(chosen);

    // dot colours: label (steps 1–2) vs outcome (step 3)
    if (onScatter) colourDots(n === 2 ? chosen : null);

    // rings on the picks (step 4)
    ringG.style.display = n === 4 ? "" : "none";
    if (n === 4) drawRings();
    if (n >= 3) colourFrontier(n === 4);

    readout.innerHTML = readoutHTML(n);
  }

  // Candidate lines: one at every label-transition midpoint on each axis.
  const xCuts = transitionMidpoints((c) => c.cheap);
  const yCuts = transitionMidpoints((c) => c.judge);
  xCuts.forEach((v) => el("line", { class: "s-cand", x1: toX(v), y1: M.top, x2: toX(v), y2: toY(0) }, candG));
  yCuts.forEach((v) => el("line", { class: "s-cand amber", x1: M.left, y1: toY(v), x2: toX(1), y2: toY(v) }, candG));

  function drawCascade(p) {
    const xW = toX(p.warn), xB = toX(p.block), yJ = toY(p.jt);
    rAllow.setAttribute("x", M.left); rAllow.setAttribute("y", M.top);
    rAllow.setAttribute("width", Math.max(0, xW - M.left)); rAllow.setAttribute("height", PH);
    rBand.setAttribute("x", xW); rBand.setAttribute("y", M.top);
    rBand.setAttribute("width", Math.max(0, xB - xW)); rBand.setAttribute("height", PH);
    rBlock.setAttribute("x", xB); rBlock.setAttribute("y", M.top);
    rBlock.setAttribute("width", Math.max(0, toX(1) - xB)); rBlock.setAttribute("height", PH);
    setLine(lWarn, xW, M.top, xW, toY(0));
    setLine(lBlock, xB, M.top, xB, toY(0));
    setLine(lJudge, xW, yJ, xB, yJ);   // judge line spans only the escalate band
  }

  function colourDots(p) {
    CASES.forEach((c, i) => {
      const d = dots[i];
      d.removeAttribute("stroke");
      if (!p) { d.setAttribute("fill", c.unsafe ? UNSAFE : SAFE); return; }
      const blocked = verdict(c, p);
      if (blocked && c.unsafe) d.setAttribute("fill", BLOCK_OK);
      else if (!blocked && !c.unsafe) d.setAttribute("fill", ALLOW_OK);
      else if (blocked) { d.setAttribute("fill", "#2a201c"); d.setAttribute("stroke", EMBER); }
      else { d.setAttribute("fill", "#241412"); d.setAttribute("stroke", DANGER); }
    });
  }

  function colourFrontier(highlight) {
    policies.forEach((p, i) => {
      const onFront = frontier.includes(p);
      polDots[i].setAttribute("fill", highlight && onFront ? TEAL_BRIGHT : (onFront ? "#4a6b60" : GREY));
      polDots[i].setAttribute("opacity", onFront ? 1 : 0.5);
    });
  }

  function drawRings() {
    while (ringG.firstChild) ringG.removeChild(ringG.firstChild);
    const entries = [["minimal", picks.minimal, TEAL], ["balanced", picks.balanced, AMBER], ["strict", picks.strict, EMBER]];
    for (const [, p, colour] of entries) {
      if (!p) continue;
      el("circle", { cx: toX(p._x), cy: toY(p._y), r: 9, fill: "none", stroke: colour, "stroke-width": 2 }, ringG);
    }
  }

  // ── side-panel readout, per step ──────────────────────────
  const unsafeN = CASES.filter((c) => c.unsafe).length;
  function readoutHTML(n) {
    if (n <= 1) {
      const cuts = xCuts.length + yCuts.length;
      return card("The dataset", [
        ["labelled cases", String(CASES.length)],
        ["unsafe / safe", `${unsafeN} / ${CASES.length - unsafeN}`],
        n === 1 ? ["candidate cuts", `${cuts} — at label flips`] : ["scored by", "2 guardrails"],
      ]);
    }
    if (n === 2) {
      const m = chosen.m;
      return card("This cascade", [
        ["screen", `block ≥ ${chosen.block.toFixed(2)} · warn ≥ ${chosen.warn.toFixed(2)}`],
        ["judge", `block ≥ ${chosen.jt.toFixed(2)}`],
        ["confusion", `tp ${m.tp} · fp ${m.fp} · tn ${m.tn} · fn ${m.fn}`],
        ["precision / recall", `${pct(m.precision)} / ${pct(m.recall)}`],
        ["F1", m.f1.toFixed(3)],
        ["judge consulted", `${pct(m.share)} of traffic`],
        ["cost", `$${m.costPer1k.toFixed(2)} / 1k`],
      ]);
    }
    if (n === 3) {
      return card("The search", [
        ["cascades scored", policies.length.toLocaleString()],
        ["method", "exhaustive — exact"],
        ["on the frontier", String(frontier.length)],
        ["dominated", String(policies.length - frontier.length)],
      ]);
    }
    return picksHTML();
  }
  function picksHTML() {
    const row = (label, p, colour) => p ? `<tr><td><span class="s-sw" style="background:${colour}"></span>${label}</td>`
      + `<td class="v">${pct(p.m.precision)}</td><td class="v">${pct(p.m.recall)}</td>`
      + `<td class="v">$${p.m.costPer1k.toFixed(2)}</td></tr>` : "";
    return `<div class="s-card"><div class="s-card-t">The three picks</div>`
      + `<table class="s-picks"><tr><th></th><th>prec</th><th>recall</th><th>$/1k</th></tr>`
      + row("Minimal", picks.minimal, TEAL) + row("Balanced", picks.balanced, AMBER) + row("Strict", picks.strict, EMBER)
      + `</table><p class="s-fine">Each ships with a 95% Wilson interval, a holdout check, and bootstrap stability.</p></div>`;
  }
  function card(title, rows) {
    return `<div class="s-card"><div class="s-card-t">${title}</div><table>`
      + rows.map(([k, v]) => `<tr><td>${k}</td><td class="v">${v}</td></tr>`).join("") + `</table></div>`;
  }

  // ── controls ──────────────────────────────────────────────
  const runBtn = document.getElementById("search-run");
  function stopAuto() { if (timer) { clearTimeout(timer); timer = 0; } runBtn.textContent = "▶ Run the search"; }
  function go(n) { stopAuto(); renderStep(clamp(n, 0, STEPS.length - 1), !reducedMotion); }
  function autoplay() {
    if (timer) { stopAuto(); return; }
    renderStep(0, !reducedMotion);
    runBtn.textContent = "⏸ pause";
    const tick = () => {
      if (step >= STEPS.length - 1) { stopAuto(); return; }
      renderStep(step + 1, !reducedMotion);
      timer = setTimeout(tick, 2600);
    };
    timer = setTimeout(tick, 2200);
  }
  runBtn.addEventListener("click", autoplay);
  document.getElementById("search-prev").addEventListener("click", () => go(step - 1));
  document.getElementById("search-next").addEventListener("click", () => go(step + 1));
  railItems.forEach((it, i) => it.addEventListener("click", () => go(i)));

  renderStep(0, false);

  // ── helpers that need CASES/policies in scope ─────────────
  function verdict(c, p) {
    if (c.cheap >= p.block) return true;
    if (c.cheap < p.warn) return false;
    return c.judge >= p.jt;
  }
  function metrics(warn, block, jt) {
    let tp = 0, fp = 0, tn = 0, fn = 0, esc = 0;
    for (const c of CASES) {
      let blocked;
      if (c.cheap >= block) blocked = true;
      else if (c.cheap < warn) blocked = false;
      else { blocked = c.judge >= jt; esc += 1; }
      if (blocked && c.unsafe) tp += 1;
      else if (blocked) fp += 1;
      else if (!c.unsafe) tn += 1;
      else fn += 1;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall = tp + fn > 0 ? tp / (tp + fn) : null;
    const fb = (b) => {
      if (precision === null || recall === null) return null;
      const b2 = b * b, d = b2 * precision + recall;
      return d > 0 ? ((1 + b2) * precision * recall) / d : null;
    };
    const share = esc / CASES.length;
    const costReq = PRICES.cheap + share * PRICES.deep;
    return { tp, fp, tn, fn, precision, recall, f05: fb(0.5), f1: fb(1), f2: fb(2),
      share, costPer1k: 1000 * costReq, costEff: 1 / costReq };
  }
  function search() {
    const G = [0.14, 0.22, 0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.86];
    const pols = [];
    for (let w = 0; w < G.length; w += 1) {
      for (let b = w + 1; b < G.length; b += 1) {
        for (let j = 0; j < G.length; j += 1) {
          const m = metrics(G[w], G[b], G[j]);
          if (m.f1 === null || m.precision === null) continue;
          pols.push({ warn: G[w], block: G[b], jt: G[j], m });
        }
      }
    }
    // Pareto frontier on (precision ↑, recall ↑) — the real optimiser's frontier: every
    // policy where you cannot raise precision without losing recall, or vice versa. (Cost
    // is a further axis; here it only breaks ties between the F-score picks below.)
    const front = pols.filter((p) => !pols.some((q) =>
      q.m.precision >= p.m.precision && q.m.recall >= p.m.recall &&
      (q.m.precision > p.m.precision + 1e-9 || q.m.recall > p.m.recall + 1e-9)));
    // Best F-score; ties go to the cheaper policy — the optimiser's own manner, and it
    // keeps the picks economical rather than escalating traffic for no accuracy gain.
    const best = (key) => front.reduce((a, p) => {
      if (a === null || p.m[key] > a.m[key] + 1e-9) return p;
      if (Math.abs(p.m[key] - a.m[key]) <= 1e-9 && p.m.costEff > a.m.costEff) return p;
      return a;
    }, null);
    return { policies: pols, frontier: front, picks: { minimal: best("f05"), balanced: best("f1"), strict: best("f2") } };
  }
  function transitionMidpoints(get) {
    const sorted = CASES.map((c) => ({ v: get(c), u: c.unsafe })).sort((a, b) => a.v - b.v);
    const cuts = new Set();
    for (let i = 1; i < sorted.length; i += 1) {
      if (sorted[i].u !== sorted[i - 1].u) cuts.add(Number(((sorted[i].v + sorted[i - 1].v) / 2).toFixed(4)));
    }
    return [...cuts];
  }
  function layoutFrontier() {
    const ps = policies.map((p) => p.m.precision), rs = policies.map((p) => p.m.recall);
    const [pLo, pHi] = [Math.min(...ps), Math.max(...ps)];
    const [rLo, rHi] = [Math.min(...rs), Math.max(...rs)];
    const norm = (v, lo, hi) => (hi - lo < 1e-9 ? 0.5 : 0.06 + 0.88 * (v - lo) / (hi - lo));
    policies.forEach((p) => { p._x = norm(p.m.precision, pLo, pHi); p._y = norm(p.m.recall, rLo, rHi); });
  }

  function setGroupOpacity(g, target, animate) {
    if (!animate || reducedMotion) { g.setAttribute("opacity", target); return; }
    const from = Number(g.getAttribute("opacity") || 1);
    if (from === target) return;
    tween({ duration: 380, ease: easeOutCubic, onUpdate: (t) => g.setAttribute("opacity", from + (target - from) * t) });
  }
}

// ── data + tiny helpers (module scope) ──────────────────────
function buildCases() {
  const rng = mulberry32(9137);
  const cs = [];
  const add = (n, unsafe, cm, csd, jm, jsd) => {
    for (let i = 0; i < n; i += 1) {
      cs.push({ unsafe, cheap: clamp(cm + gaussian(rng) * csd, 0.02, 0.98),
        judge: clamp(jm + gaussian(rng) * jsd, 0.02, 0.98) });
    }
  };
  // The extremes sit well outside any escalation band, so the cheap screen clears them for
  // free; the uncertain middle overlaps on BOTH axes, so no cascade separates it perfectly
  // — that residual overlap is the real precision/recall trade the profiles divide.
  add(36, false, 0.18, 0.07, 0.20, 0.08);   // clearly safe — the screen allows outright
  add(22, true, 0.85, 0.06, 0.82, 0.08);     // clearly unsafe — the screen blocks outright
  add(18, false, 0.50, 0.10, 0.44, 0.15);    // uncertain middle: the judge only leans low
  add(18, true, 0.55, 0.10, 0.58, 0.15);     // uncertain middle: the judge only leans high
  add(8, false, 0.64, 0.08, 0.30, 0.11);     // screen-risky, judge-fine — the band clears it
  return cs;
}
const pct = (v) => (v === null ? "—" : (v * 100).toFixed(0) + "%");
function setLine(node, x1, y1, x2, y2) {
  node.setAttribute("x1", x1); node.setAttribute("y1", y1);
  node.setAttribute("x2", x2); node.setAttribute("y2", y2);
}
