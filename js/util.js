// Shared helpers. Deterministic randomness everywhere: the page must look the same on
// every load, or a screenshot in a talk stops matching the site.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  // Box–Muller; one value per call is plenty here.
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function tween({ duration, ease = easeOutCubic, onUpdate, onDone }) {
  const start = performance.now();
  function frame(now) {
    const t = clamp((now - start) / duration, 0, 1);
    onUpdate(ease(t));
    if (t < 1) requestAnimationFrame(frame);
    else if (onDone) onDone();
  }
  requestAnimationFrame(frame);
}

export const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

export function money(value, digits = 2) {
  return "$" + value.toFixed(digits);
}

// Real measured per-call prices for the three-tier judge fleet (see the paper notes:
// gemini-flash-lite / haiku / sonnet ≈ 1× / 21× / 52×).
export const PRICES = { cheap: 0.000031, mid: 0.000661, deep: 0.001607 };

export const svgNS = "http://www.w3.org/2000/svg";
export function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}
