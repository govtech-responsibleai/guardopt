// A2 — the crystallisation. The search itself, live in your browser.
//
// This is not a canned animation of a search: the 21,952 policies are genuinely
// enumerated and evaluated here, with the same semantics as the real optimiser's flat
// path — verdict = OR over enabled guardrails, a policy that blocks nothing has
// undefined precision and is EXCLUDED rather than scored zero. The evaluation is
// bitmask arithmetic (300 cases = ten 32-bit words per mask), so the honest compute
// time is tens of milliseconds; the bloom is slowed down for drama and the status
// line says both numbers, because pretending the reveal is the compute would be a
// tiny lie on a page about not lying.

import { mulberry32, gaussian, clamp, PRICES } from "./util.js";

const CASES = 300;
const WORDS = Math.ceil(CASES / 32);
const THRESHOLDS = 27; // + "off" per guardrail → 28^3 = 21,952 policies

const GUARDRAILS = [
  { name: "screen", price: PRICES.cheap, unsafeMean: 0.66, safeMean: 0.38, sd: 0.2 },
  { name: "mid", price: PRICES.mid, unsafeMean: 0.72, safeMean: 0.3, sd: 0.15 },
  { name: "judge", price: PRICES.deep, unsafeMean: 0.8, safeMean: 0.22, sd: 0.09 },
];

function popcount(x) {
  x -= (x >> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >> 24;
}

function evaluateEverything() {
  const rng = mulberry32(90210);
  const unsafe = new Uint8Array(CASES);
  const scores = GUARDRAILS.map(() => new Float64Array(CASES));
  for (let c = 0; c < CASES; c += 1) {
    unsafe[c] = rng() < 0.3 ? 1 : 0;
    for (let g = 0; g < 3; g += 1) {
      const spec = GUARDRAILS[g];
      const mean = unsafe[c] ? spec.unsafeMean : spec.safeMean;
      scores[g][c] = clamp(mean + gaussian(rng) * spec.sd, 0.01, 0.99);
    }
  }

  const unsafeMask = new Uint32Array(WORDS);
  let unsafeCount = 0;
  for (let c = 0; c < CASES; c += 1) if (unsafe[c]) {
    unsafeMask[c >> 5] |= 1 << (c & 31);
    unsafeCount += 1;
  }
  const validMask = new Uint32Array(WORDS);
  for (let c = 0; c < CASES; c += 1) validMask[c >> 5] |= 1 << (c & 31);

  // Fire masks: guardrail g at threshold i blocks case c iff score >= t.
  const fire = GUARDRAILS.map((_, g) => {
    const perThreshold = [];
    for (let i = 0; i < THRESHOLDS; i += 1) {
      const t = 0.02 + (0.96 * i) / (THRESHOLDS - 1);
      const mask = new Uint32Array(WORDS);
      for (let c = 0; c < CASES; c += 1) if (scores[g][c] >= t) mask[c >> 5] |= 1 << (c & 31);
      perThreshold.push(mask);
    }
    return perThreshold;
  });

  // The space: every flat OR-combination of the three guardrails (28^3 including
  // "off" slots), PLUS every two-threshold cascade — screen band (warn < block)
  // routing the uncertain middle to the judge. Cascade cost varies continuously with
  // the escalated share, which is exactly why the frontier is worth drawing at all:
  // flat policies can only occupy eight cost columns (one per guardrail subset).
  const flatTotal = Math.pow(THRESHOLDS + 1, 3);
  const bandPairs = (THRESHOLDS * (THRESHOLDS - 1)) / 2;
  const total = flatTotal + bandPairs * THRESHOLDS;
  const cost = new Float64Array(total);
  const f05 = new Float64Array(total);
  const f1 = new Float64Array(total);
  const f2 = new Float64Array(total);
  const valid = new Uint8Array(total);
  const kind = new Uint8Array(total); // 0 flat, 1 cascade
  const combined = new Uint32Array(WORDS);

  const started = performance.now();
  let index = 0;
  let excluded = 0;

  const score = (blockedMask, policyCost, policyKind) => {
    let tp = 0, blocked = 0;
    for (let w = 0; w < WORDS; w += 1) {
      const m = blockedMask[w] & validMask[w];
      blocked += popcount(m);
      tp += popcount(m & unsafeMask[w]);
    }
    if (blocked === 0) { excluded += 1; index += 1; return; }
    const precision = tp / blocked;
    const recall = tp / unsafeCount;
    const fBeta = (beta) => {
      const b2 = beta * beta;
      const den = b2 * precision + recall;
      return den === 0 ? 0 : ((1 + b2) * precision * recall) / den;
    };
    valid[index] = 1;
    kind[index] = policyKind;
    f05[index] = fBeta(0.5);
    f1[index] = fBeta(1);
    f2[index] = fBeta(2);
    cost[index] = policyCost;
    index += 1;
  };

  for (let a = -1; a < THRESHOLDS; a += 1) {
    for (let b = -1; b < THRESHOLDS; b += 1) {
      for (let c = -1; c < THRESHOLDS; c += 1) {
        combined.fill(0);
        if (a >= 0) for (let w = 0; w < WORDS; w += 1) combined[w] |= fire[0][a][w];
        if (b >= 0) for (let w = 0; w < WORDS; w += 1) combined[w] |= fire[1][b][w];
        if (c >= 0) for (let w = 0; w < WORDS; w += 1) combined[w] |= fire[2][c][w];
        score(combined,
          (a >= 0 ? GUARDRAILS[0].price : 0)
            + (b >= 0 ? GUARDRAILS[1].price : 0)
            + (c >= 0 ? GUARDRAILS[2].price : 0),
          0);
      }
    }
  }

  // Cascades: screen fires-block at index b, escalates in [w, b), judge adjudicates
  // the band at threshold j. Thresholds ascend with index, so w < b means the warn
  // line sits below the block line — the band is fire[w] minus fire[b].
  const band = new Uint32Array(WORDS);
  for (let w = 0; w < THRESHOLDS; w += 1) {
    for (let b = w + 1; b < THRESHOLDS; b += 1) {
      let escalated = 0;
      for (let word = 0; word < WORDS; word += 1) {
        band[word] = fire[0][w][word] & ~fire[0][b][word];
        escalated += popcount(band[word] & validMask[word]);
      }
      const cascadeCost = GUARDRAILS[0].price
        + (escalated / CASES) * GUARDRAILS[2].price;
      for (let j = 0; j < THRESHOLDS; j += 1) {
        for (let word = 0; word < WORDS; word += 1) {
          combined[word] = fire[0][b][word] | (band[word] & fire[2][j][word]);
        }
        score(combined, cascadeCost, 1);
      }
    }
  }

  const computeMs = performance.now() - started;
  return { total, cost, f05, f1, f2, valid, kind, excluded, computeMs };
}

export function initFrontier(reducedMotion) {
  const canvas = document.getElementById("frontier-canvas");
  const ctx = canvas.getContext("2d");
  const status = document.getElementById("frontier-status");
  const runButton = document.getElementById("frontier-run");

  let W = 0, H = 0, dpr = 1;
  const M = { left: 78, right: 26, top: 26, bottom: 56 };
  let results = null;
  let phase = "idle"; // idle | blooming | done
  let layers = null;

  // The x axis is COST EFFECTIVENESS — requests served per dollar — so both axes
  // read the same way: up and right is better. Log scale; the fleet spans ~435
  // req/$ (everything through the frontier judge) to ~32k req/$ (screen only).
  const costMin = PRICES.cheap * 0.8;
  const costMax = (PRICES.cheap + PRICES.mid + PRICES.deep) * 1.15;
  const xOf = (c) => M.left
    + ((Math.log10(costMax) - Math.log10(c)) / (Math.log10(costMax) - Math.log10(costMin)))
      * (W - M.left - M.right);
  const yOf = (f) => M.top + (1 - f) * (H - M.top - M.bottom);

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssWidth = canvas.clientWidth || canvas.parentElement.clientWidth;
    const cssHeight = Math.max(300, Math.round(cssWidth * 0.5));
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.height = cssHeight + "px";
    W = cssWidth; H = cssHeight;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (phase === "idle") drawChrome("press evaluate — the search runs in this tab, not on a server");
    if (phase === "done" && results) drawFinal();
  }

  function drawChrome(hint) {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#243029";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(M.left, M.top); ctx.lineTo(M.left, H - M.bottom); ctx.lineTo(W - M.right, H - M.bottom);
    ctx.stroke();
    ctx.fillStyle = "#5e6f69";
    ctx.font = "11px ui-monospace, Menlo, monospace";
    for (const f of [0.25, 0.5, 0.75, 1]) {
      ctx.fillText(f.toFixed(2), M.left - 34, yOf(f) + 4);
      ctx.strokeStyle = "rgba(36,48,41,0.5)";
      ctx.beginPath(); ctx.moveTo(M.left, yOf(f)); ctx.lineTo(W - M.right, yOf(f)); ctx.stroke();
    }
    for (const [c, label] of [[0.001, "1k req/$"], [0.0001, "10k req/$"]]) {
      ctx.fillText(label, xOf(c) - 24, H - M.bottom + 18);
    }
    ctx.fillText("cost effectiveness (requests per dollar, log) → better", W / 2 - 150, H - M.bottom + 36);
    ctx.save();
    ctx.translate(22, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("F1 →", -14, 0);
    ctx.restore();
    if (hint) {
      ctx.fillStyle = "#5e6f69";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(hint, M.left + 24, M.top + 28);
    }
  }

  function dot(context, x, y, r, fill) {
    context.fillStyle = fill;
    context.beginPath();
    context.arc(x, y, r, 0, Math.PI * 2);
    context.fill();
  }

  function makeLayer(draw) {
    const layer = document.createElement("canvas");
    layer.width = canvas.width; layer.height = canvas.height;
    const layerCtx = layer.getContext("2d");
    layerCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(layerCtx);
    return layer;
  }

  function frontierIndices() {
    const order = [];
    for (let i = 0; i < results.total; i += 1) if (results.valid[i]) order.push(i);
    order.sort((p, q) => results.cost[p] - results.cost[q] || results.f1[q] - results.f1[p]);
    const frontier = [];
    let best = -1;
    for (const i of order) {
      if (results.f1[i] > best + 1e-12) { frontier.push(i); best = results.f1[i]; }
    }
    return frontier;
  }

  function argmax(array) {
    let bestIndex = -1, bestValue = -1;
    for (let i = 0; i < results.total; i += 1) {
      if (results.valid[i] && array[i] > bestValue) { bestValue = array[i]; bestIndex = i; }
    }
    return bestIndex;
  }

  function drawFinal() {
    drawChrome();
    const frontier = new Set(frontierIndices());
    for (let i = 0; i < results.total; i += 1) {
      if (!results.valid[i]) continue;
      if (!frontier.has(i)) dot(ctx, xOf(results.cost[i]), yOf(results.f1[i]), 2.2, "rgba(94,111,105,0.25)");
    }
    drawFrontier(1);
  }

  function drawFrontier(t) {
    const frontier = frontierIndices().reverse(); // left-to-right on the flipped axis
    ctx.strokeStyle = "rgba(127,227,204,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const visible = Math.max(2, Math.floor(frontier.length * t));
    frontier.slice(0, visible).forEach((i, k) => {
      const x = xOf(results.cost[i]), y = yOf(results.f1[i]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    for (const i of frontier.slice(0, visible)) {
      dot(ctx, xOf(results.cost[i]), yOf(results.f1[i]), 3.6, "#7fe3cc");
    }
    if (t >= 1) {
      const picks = [
        [argmax(results.f05), "Minimal · F0.5"],
        [argmax(results.f1), "Balanced · F1"],
        [argmax(results.f2), "Strict · F2"],
      ];
      ctx.font = "12px ui-monospace, Menlo, monospace";
      const placed = [];
      for (const [i, label] of picks) {
        if (i < 0) continue;
        const x = xOf(results.cost[i]), y = yOf(results.f1[i]);
        ctx.strokeStyle = "#e9efec";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
        let ly = y - 14;
        while (placed.some((p) => Math.abs(p - ly) < 14)) ly -= 15;
        placed.push(ly);
        ctx.fillStyle = "#e9efec";
        ctx.fillText(label, Math.min(x + 14, W - 130), ly);
      }
    }
  }

  function run() {
    runButton.disabled = true;
    status.textContent = "evaluating…";
    // Yield one frame so the button state paints before the synchronous burst.
    requestAnimationFrame(() => {
      results = evaluateEverything();
      const validIndices = [];
      for (let i = 0; i < results.total; i += 1) if (results.valid[i]) validIndices.push(i);
      // Shuffle deterministically for an even bloom.
      const rng = mulberry32(7);
      for (let i = validIndices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(rng() * (i + 1));
        [validIndices[i], validIndices[j]] = [validIndices[j], validIndices[i]];
      }

      if (reducedMotion) {
        phase = "done";
        drawFinal();
        finishStatus();
        runButton.disabled = false;
        runButton.textContent = "↻ replay";
        return;
      }

      phase = "blooming";
      drawChrome();
      const BLOOM_MS = 1700;
      const start = performance.now();
      let shown = 0;

      const bloom = (now) => {
        const t = Math.min((now - start) / BLOOM_MS, 1);
        const target = Math.floor(t * t * validIndices.length);
        while (shown < target) {
          const i = validIndices[shown];
          dot(ctx, xOf(results.cost[i]), yOf(results.f1[i]), 2.4, "rgba(91,196,173,0.4)");
          shown += 1;
        }
        status.textContent = `evaluated ${(shown + results.excluded).toLocaleString()} / ${results.total.toLocaleString()}`
          + ` · real compute: ${results.computeMs.toFixed(0)} ms`;
        if (t < 1) requestAnimationFrame(bloom);
        else crystallise(validIndices);
      };
      requestAnimationFrame(bloom);
    });
  }

  function crystallise(validIndices) {
    const frontier = new Set(frontierIndices());
    const greyLayer = makeLayer((layerCtx) => {
      for (const i of validIndices) {
        if (!frontier.has(i)) dot(layerCtx, xOf(results.cost[i]), yOf(results.f1[i]), 2.2, "rgba(94,111,105,0.25)");
      }
    });
    const tealLayer = makeLayer((layerCtx) => {
      for (const i of validIndices) {
        dot(layerCtx, xOf(results.cost[i]), yOf(results.f1[i]), 2.4, "rgba(91,196,173,0.4)");
      }
    });
    layers = { greyLayer, tealLayer };

    const FADE_MS = 800, LINE_MS = 700;
    const start = performance.now();
    const phase2 = (now) => {
      const elapsed = now - start;
      const fade = Math.min(elapsed / FADE_MS, 1);
      drawChrome();
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1 - fade;
      ctx.drawImage(layers.tealLayer, 0, 0);
      ctx.globalAlpha = fade;
      ctx.drawImage(layers.greyLayer, 0, 0);
      ctx.restore();
      const line = clamp((elapsed - FADE_MS * 0.6) / LINE_MS, 0, 1);
      if (line > 0) drawFrontier(line);
      if (fade < 1 || line < 1) requestAnimationFrame(phase2);
      else {
        phase = "done";
        finishStatus();
        runButton.disabled = false;
        runButton.textContent = "↻ replay";
      }
    };
    requestAnimationFrame(phase2);
  }

  function finishStatus() {
    const frontier = frontierIndices();
    const cascades = frontier.filter((i) => results.kind[i] === 1).length;
    status.textContent = `${results.total.toLocaleString()} policies · ${results.excluded.toLocaleString()} excluded`
      + ` (blocked nothing — precision undefined, not zero) · frontier: ${frontier.length}`
      + ` (${cascades} cascades) · real compute: ${results.computeMs.toFixed(0)} ms`;
  }

  runButton.addEventListener("click", run);
  window.addEventListener("resize", resize);
  resize();
  // Sharable auto-play link (also what the smoke test drives): /?run#frontier
  if (new URLSearchParams(window.location.search).has("run")) run();
}
