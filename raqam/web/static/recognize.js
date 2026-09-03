/* Offline recognizer — pure JS, no WASM, no deps.
 * Mirrors raqam/cnn.py forward + raqam/segment.py cell extraction so the PWA
 * digitizes a form field with zero connectivity (plan §03). */

let MODEL = null;

export async function loadModel(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('model fetch failed');
  const j = await r.json();
  MODEL = {
    W1: Float32Array.from(j.W1), b1: Float32Array.from(j.b1),
    W2: Float32Array.from(j.W2), b2: Float32Array.from(j.b2),
    W3: Float32Array.from(j.W3), b3: Float32Array.from(j.b3),
    W4: Float32Array.from(j.W4), b4: Float32Array.from(j.b4),
    shapes: j.shapes, temperature: j.temperature || 1.7,
  };
  return MODEL;
}
export const modelReady = () => MODEL !== null;

function conv(src, Cin, H, W, wt, bias, Cout) {           // same padding, 3x3, stride 1
  const out = new Float32Array(Cout * H * W);
  for (let oc = 0; oc < Cout; oc++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = bias[oc];
      for (let ic = 0; ic < Cin; ic++) {
        for (let ky = 0; ky < 3; ky++) {
          const iy = y + ky - 1; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = x + kx - 1; if (ix < 0 || ix >= W) continue;
            acc += src[(ic * H + iy) * W + ix] *
                   wt[((oc * Cin + ic) * 3 + ky) * 3 + kx];
          }
        }
      }
      out[(oc * H + y) * W + x] = acc > 0 ? acc : 0;         // conv + ReLU
    }
  }
  return out;
}

function pool2(src, C, H, W) {
  const oh = H >> 1, ow = W >> 1, out = new Float32Array(C * oh * ow);
  for (let c = 0; c < C; c++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    let m = -Infinity;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++)
      m = Math.max(m, src[(c * H + 2 * y + dy) * W + 2 * x + dx]);
    out[(c * oh + y) * ow + x] = m;
  }
  return out;
}

/* px28: Float32Array length 784, values 0..1 -> {digit, conf, probs} */
export function classify(px28, temperature) {
  if (!MODEL) throw new Error('model not loaded');
  temperature = temperature || MODEL.temperature || 1.7;
  let h = conv(px28, 1, 28, 28, MODEL.W1, MODEL.b1, 8);
  h = pool2(h, 8, 28, 28);
  h = conv(h, 8, 14, 14, MODEL.W2, MODEL.b2, 16);
  h = pool2(h, 16, 14, 14);                                  // 16 x 7 x 7 = 784
  const fc = 64;
  const r3 = new Float32Array(fc);
  for (let j = 0; j < fc; j++) {
    let a = MODEL.b3[j];
    for (let i = 0; i < 784; i++) a += h[i] * MODEL.W3[i * fc + j];
    r3[j] = a > 0 ? a : 0;
  }
  const logit = new Float32Array(10);
  for (let k = 0; k < 10; k++) {
    let a = MODEL.b4[k];
    for (let j = 0; j < fc; j++) a += r3[j] * MODEL.W4[j * 10 + k];
    logit[k] = a / temperature;
  }
  let mx = -Infinity; for (const v of logit) mx = Math.max(mx, v);
  let sum = 0; const probs = logit.map(v => { const e = Math.exp(v - mx); sum += e; return e; });
  for (let k = 0; k < 10; k++) probs[k] /= sum;
  let digit = 0; for (let k = 1; k < 10; k++) if (probs[k] > probs[digit]) digit = k;
  return { digit, conf: probs[digit], probs: Array.from(probs) };
}

/* ---- segmentation: photo -> printed digit boxes -> 28x28 cells ---- */
function otsu(gray) {
  const hist = new Array(256).fill(0);
  for (const v of gray) hist[v]++;
  const total = gray.length;
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, thr = 127;
  for (let i = 0; i < 256; i++) {
    wB += hist[i]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; thr = i; }
  }
  return thr;
}

/* returns {cells: [Float32Array(784)], boxes: [{x,y,w,h}], w, h, gray } from an ImageData */
export function segment(imgData) {
  const { width: W, height: H, data } = imgData;
  const gray = new Uint8ClampedArray(W * H);
  for (let i = 0; i < W * H; i++)
    gray[i] = (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114) | 0;
  const thr = otsu(gray);
  const ink = new Uint8Array(W * H);                          // 1 = dark (ink/border)
  for (let i = 0; i < W * H; i++) ink[i] = gray[i] < thr ? 1 : 0;

  // connected components (4-conn) over ink, flood fill
  const lab = new Int32Array(W * H).fill(0);
  const comps = [];
  const stack = [];
  let next = 1;
  for (let p = 0; p < W * H; p++) {
    if (!ink[p] || lab[p]) continue;
    stack.length = 0; stack.push(p); lab[p] = next;
    let minx = W, miny = H, maxx = 0, maxy = 0, area = 0;
    while (stack.length) {
      const q = stack.pop(), qx = q % W, qy = (q / W) | 0;
      area++;
      if (qx < minx) minx = qx; if (qx > maxx) maxx = qx;
      if (qy < miny) miny = qy; if (qy > maxy) maxy = qy;
      if (qx > 0 && ink[q - 1] && !lab[q - 1]) { lab[q - 1] = next; stack.push(q - 1); }
      if (qx < W - 1 && ink[q + 1] && !lab[q + 1]) { lab[q + 1] = next; stack.push(q + 1); }
      if (qy > 0 && ink[q - W] && !lab[q - W]) { lab[q - W] = next; stack.push(q - W); }
      if (qy < H - 1 && ink[q + W] && !lab[q + W]) { lab[q + W] = next; stack.push(q + W); }
    }
    comps.push({ x: minx, y: miny, w: maxx - minx + 1, h: maxy - miny + 1, area });
    next++;
  }

  const imgArea = W * H;
  // a printed cell is a hollow frame: bbox large, ink pixels ≈ perimeter (fill ratio low)
  let boxes = comps.filter(c => {
    const a = c.w * c.h, ar = c.w / c.h, fill = c.area / a;
    return a > 0.004 * imgArea && a < 0.22 * imgArea
      && ar > 0.3 && ar < 3.2 && fill > 0.02 && fill < 0.65;
  }).sort((p, q) => q.w * q.h - p.w * p.h);
  // drop any component whose centre sits inside a larger kept box (a digit inside its cell)
  const kept = [];
  for (const b of boxes) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    if (kept.some(k => cx > k.x - 4 && cx < k.x + k.w + 4 && cy > k.y - 4 && cy < k.y + k.h + 4
        && b.w * b.h < 0.9 * k.w * k.h)) continue;
    kept.push(b);
  }
  // keep only cells near the dominant size (reject stray marks / text)
  const areas = kept.map(b => b.w * b.h).sort((a, b) => a - b);
  const medA = areas[areas.length >> 1] || 1;
  boxes = kept.filter(b => { const a = b.w * b.h; return a > 0.4 * medA && a < 2.5 * medA; });
  if (!boxes.length) return { cells: [], boxes: [], w: W, h: H, gray };
  const medH = boxes.map(b => b.h).sort((a, b) => a - b)[boxes.length >> 1];
  boxes.sort((a, b) => (Math.round(a.y / (medH * 0.7)) - Math.round(b.y / (medH * 0.7))) || a.x - b.x);

  // extract, then keep only boxes that actually contain a plausible glyph:
  // interior ink fraction between 0.4% and 45% (empty boxes and solid blobs out)
  const out = [];
  for (const b of boxes) {
    const { px, ink } = cellToMnist(gray, W, H, b, thr);
    if (ink > 0.004 && ink < 0.45) out.push({ box: b, px });
  }
  // a lone box with weak evidence is almost always background noise, not a field
  if (out.length === 1 && !hasInkRun(gray, W, thr, out[0].box)) {
    return { cells: [], boxes: [], w: W, h: H, gray };
  }
  return { cells: out.map(o => o.px), boxes: out.map(o => o.box), w: W, h: H, gray };
}

// is there a horizontal run of ink (a digit strip) roughly where this box sits?
function hasInkRun(gray, W, thr, box) {
  const y = Math.round(box.y + box.h / 2);
  let run = 0, best = 0;
  for (let x = box.x - box.w; x < box.x + 2 * box.w; x++) {
    if (x < 0 || x >= W) continue;
    if (gray[y * W + x] < thr) { run++; best = Math.max(best, run); } else run = 0;
  }
  return best > box.w * 0.15;
}

function cellToMnist(gray, W, H, box, thr) {
  const pad = Math.round(Math.min(box.w, box.h) * 0.16);
  const x0 = box.x + pad, y0 = box.y + pad;
  const cw = box.w - 2 * pad, ch = box.h - 2 * pad;
  const EMPTY = { px: new Float32Array(784), ink: 0 };
  if (cw < 4 || ch < 4) return EMPTY;
  let minx = cw, miny = ch, maxx = 0, maxy = 0, inkPx = 0;
  const local = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const g = gray[(y0 + y) * W + (x0 + x)];
    const v = g < thr ? (thr - g) / thr : 0;
    local[y * cw + x] = v;
    if (v > 0.15) { inkPx++; if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; }
  }
  const inkFrac = inkPx / (cw * ch);
  if (maxx <= minx) return EMPTY;
  const gw = maxx - minx + 1, gh = maxy - miny + 1;
  const s = 20 / Math.max(gw, gh);
  const dw = Math.max(1, Math.round(gw * s)), dh = Math.max(1, Math.round(gh * s));
  const out = new Float32Array(784);
  const ox = ((28 - dw) / 2) | 0, oy = ((28 - dh) / 2) | 0;
  for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
    const sx = minx + Math.min(gw - 1, (x / s) | 0), sy = miny + Math.min(gh - 1, (y / s) | 0);
    out[(oy + y) * 28 + (ox + x)] = local[sy * cw + sx];
  }
  // centre of mass -> (14,14)
  let mx = 0, my = 0, tot = 0;
  for (let i = 0; i < 784; i++) { mx += (i % 28) * out[i]; my += ((i / 28) | 0) * out[i]; tot += out[i]; }
  if (tot > 0) {
    const shx = Math.round(14 - mx / tot), shy = Math.round(14 - my / tot);
    if (shx || shy) {
      const shifted = new Float32Array(784);
      for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
        const nx = x + shx, ny = y + shy;
        if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) shifted[ny * 28 + nx] = out[y * 28 + x];
      }
      return { px: shifted, ink: inkFrac };
    }
  }
  return { px: out, ink: inkFrac };
}
