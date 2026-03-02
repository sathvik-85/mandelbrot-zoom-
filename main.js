import init from './pkg/mandelbrot_wasm.js';
import { GlRenderer } from './gl-renderer.js';

async function run() {
  await init();

  const canvas   = document.getElementById('canvas');
  const ctx      = canvas.getContext('2d');
  const glCanvas = document.getElementById('canvas-gl');
  const evt      = document.getElementById('evt');
  const hud      = document.getElementById('coords');

  let W = canvas.width  = glCanvas.width  = window.innerWidth;
  let H = canvas.height = glCanvas.height = window.innerHeight;

  // ── WebGL renderer (GPU path, used when zoom < GL_ZOOM_LIMIT) ────────────
  const GL_ZOOM_LIMIT = 5e10;
  let glr = null;
  let useGL = false;
  try {
    glr = new GlRenderer(glCanvas);
    glr.resize(W, H);
    useGL = true;
  } catch (e) {
    console.warn('WebGL2 unavailable, using WASM only:', e.message);
  }

  // Show the right canvas based on mode
  function applyMode() {
    canvas.style.display   = useGL ? 'none'  : 'block';
    glCanvas.style.display = useGL ? 'block' : 'none';
  }
  applyMode();

  // GL dirty-render: coalesce multiple event calls into one RAF frame
  let glDirty = false;
  let glRafId = null;
  function markGlDirty() {
    glDirty = true;
    if (!glRafId) glRafId = requestAnimationFrame(glFrame);
  }
  function glFrame() {
    glRafId = null;
    if (!useGL || !glDirty) return;
    glDirty = false;
    glr.render(cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter(), W, H, pal);
  }

  // Check whether to switch between GL and WASM modes based on current zoom
  function checkMode() {
    const wantGL = glr !== null && zoom < GL_ZOOM_LIMIT;
    if (wantGL === useGL) return;
    useGL = wantGL;
    applyMode();
    resetCSS();
    if (useGL) {
      // Cancel any pending WASM timers so they don't fire on the hidden canvas
      clearTimeout(previewTimer);
      clearTimeout(settleTimer);
      showLoading(false);
      markGlDirty();
    } else {
      renderParallel();
    }
  }

  // ── Worker pool ───────────────────────────────────────────────────────────
  const NUM_WORKERS = Math.min(navigator.hardwareConcurrency || 4, 8);
  const workers = Array.from({ length: NUM_WORKERS }, () =>
    new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
  );
  let jobId = 0;
  let rendering = false;

  // ── Mandelbrot state ─────────────────────────────────────────────────────
  // Center stored as double-double (hi + lo) for ~32 digits of precision.
  // Extends zoom from f64 limit (~1e13) to ~1e26 without visible pixelation.
  let cx_hi = -0.5, cx_lo = 0.0;
  let cy_hi =  0.0, cy_lo = 0.0;
  let zoom  =  1.0;
  let pal   =  0;   // 0=Nebula, 1=Fire, 2=Ice

  // Two-sum: exact split of (a + b) into (sum, error)
  function twoSum(a, b) {
    const s = a + b, v = s - a;
    return [s, (a - (s - v)) + (b - v)];
  }
  // Add a plain f64 delta to a DD accumulator with full precision
  function ddAdd(hi, lo, delta) {
    const [s, e] = twoSum(hi, delta);
    return [s, e + lo];
  }

  // During preset fly-in animations we cap iterations low so the GPU keeps 60 fps.
  let animating = false;

  function maxIter() {
    const iter = Math.min(Math.floor(80 + 15 * Math.log2(zoom + 1)), 1200);
    return animating ? Math.min(iter, 120) : iter;
  }

  // Loading indicator — shown only during WASM (CPU) renders
  function showLoading(on) {
    document.getElementById('loading').classList.toggle('hidden', !on);
  }

  // ── CSS transform: instant GPU visual feedback, zero compute ──────────────
  // The canvas always shows the last true render. During interaction we CSS-
  // scale/translate it. When new render is ready we snap back to identity.
  canvas.style.transformOrigin = '0 0';
  let vScale = 1, vTx = 0, vTy = 0; // visual offset from last render

  function applyCSS() {
    canvas.style.transform =
      `matrix(${vScale}, 0, 0, ${vScale}, ${vTx}, ${vTy})`;
  }
  function resetCSS() {
    vScale = 1; vTx = 0; vTy = 0;
    canvas.style.transform = '';
  }

  // ── Progressive parallel render ───────────────────────────────────────────
  // Two-pass strategy:
  //   Pass 1 (quality=0.25): renders W/4 × H/4 pixels → drawn scaled up.
  //                          ~16× fewer pixels → completes in ~20–50 ms.
  //   Pass 2 (quality=1.0):  full resolution, replaces the blurry preview.
  // During animation only pass 1 fires so we never fall behind the camera.

  let previewTimer = null;
  let settleTimer  = null;

  // Used for animation events and palette/resize — fires renderParallel after `delay` ms.
  function scheduleRender(delay = 0, fullRes = true) {
    clearTimeout(previewTimer);
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => renderParallel(fullRes), delay);
  }

  // Used for scroll / drag / touch — fires a quick 25 % preview at 50 ms idle,
  // then the full 3-pass render at 250 ms idle. Much more responsive than a
  // single 150 ms debounce that shows nothing in between.
  function scheduleInteractionRender() {
    clearTimeout(previewTimer);
    clearTimeout(settleTimer);
    previewTimer = setTimeout(() => {
      if (rendering) jobId++;
      rendering = true;
      renderAtQuality(0.25, job => { if (job === jobId) rendering = false; });
    }, 50);
    settleTimer = setTimeout(() => renderParallel(), 250);
  }

  // Render at a fractional resolution and call onDone(jobId) when all strips land.
  function renderAtQuality(quality, onDone) {
    const thisJob = ++jobId;
    const rW      = Math.max(Math.round(W * quality), 2);
    const rH      = Math.max(Math.round(H * quality), 2);
    const stripH  = Math.ceil(rH / NUM_WORKERS);
    const results = new Array(NUM_WORKERS).fill(null);
    let done = 0;

    workers.forEach((worker, i) => {
      const yStart = i * stripH;
      const yEnd   = Math.min(yStart + stripH, rH);

      // Replacing onmessage cancels the previous pass for this worker.
      worker.onmessage = ({ data }) => {
        if (data.jobId !== thisJob) return; // stale — discard
        results[i] = data;
        if (++done < NUM_WORKERS) return;

        if (quality < 1.0) {
          // Composite strips into a small offscreen canvas, then scale to screen.
          const off    = new OffscreenCanvas(rW, rH);
          const offCtx = off.getContext('2d');
          results.forEach(r => {
            if (!r) return;
            offCtx.putImageData(
              new ImageData(new Uint8ClampedArray(r.pixels), rW, r.yEnd - r.yStart),
              0, r.yStart,
            );
          });
          ctx.imageSmoothingEnabled  = true;
          ctx.imageSmoothingQuality  = 'high';
          ctx.drawImage(off, 0, 0, W, H);
        } else {
          // Full-res — write each strip directly into the canvas.
          results.forEach(r => {
            if (!r) return;
            ctx.putImageData(
              new ImageData(new Uint8ClampedArray(r.pixels), W, r.yEnd - r.yStart),
              0, r.yStart,
            );
          });
        }

        resetCSS();
        if (onDone) onDone(thisJob);
      };

      worker.postMessage({
        cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter: maxIter(),
        canvasW: rW, canvasH: rH, yStart, yEnd, jobId: thisJob, pal,
      });
    });
  }

  function updateHUD() {
    hud.textContent =
      `x: ${cx_hi.toFixed(10)}  y: ${cy_hi.toFixed(10)}  zoom: ${zoom.toExponential(2)}  iter: ${maxIter()}  cores: ${NUM_WORKERS}`;
  }

  function renderParallel(doFullRes = true) {
    if (rendering) jobId++; // invalidate any in-flight strips
    rendering = true;
    showLoading(true);

    // Pass 1 — instant blurry preview (~5–15 ms)
    renderAtQuality(0.25, (job1) => {
      if (!doFullRes || job1 !== jobId) return;

      // Pass 2 — medium preview (~50–150 ms), much less blurry
      renderAtQuality(0.5, (job2) => {
        if (job2 !== jobId) return;

        // Pass 3 — crisp full resolution
        renderAtQuality(1.0, () => {
          rendering = false;
          showLoading(false);
          updateHUD();
        });
      });
    });
  }

  // ── Famous locations ──────────────────────────────────────────────────────
  const PRESETS = [
    { cx: -0.5,            cy:  0.0,           zoom:      1, label: 'Home'            },
    { cx: -0.7436438870,   cy:  0.1318259042,  zoom:   8000, label: 'Seahorse Valley' },
    { cx: -0.7269,         cy:  0.1889,        zoom:    500, label: 'Double Spiral'   },
    { cx: -1.7498,         cy:  0.0,           zoom:   4000, label: 'Mini Mandelbrot' },
    { cx: -0.1592,         cy:  1.0317,        zoom:    800, label: 'Triple Spiral'   },
    { cx:  0.2800,         cy:  0.0085,        zoom:   2000, label: 'Elephant Valley' },
    { cx: -1.401155189,    cy:  0.0,           zoom:  80000, label: 'Feigenbaum Point'},
    { cx: -1.755,          cy:  0.0,           zoom:    300, label: 'Period-3 Bulb'   },
    { cx: -0.10109636,     cy:  0.95628651,    zoom:   8000, label: 'Misiurewicz'     },
    { cx: -0.7453054,      cy:  0.1125006,     zoom:  60000, label: 'Spiral Valley'   },
    { cx: -0.8100,         cy:  0.1560,        zoom:   3000, label: 'Star Cluster'    },
    { cx: -0.7629,         cy:  0.0898,        zoom:  40000, label: 'Deep Valley'     },
    { cx: -0.5591,         cy:  0.6355,        zoom:  10000, label: "Dragon's Tail"   },
    { cx: -0.1248,         cy:  0.8439,        zoom:   6000, label: 'Quad Spiral'     },
    { cx: -1.9431,         cy:  0.0,           zoom:   1000, label: 'Baby Mandelbrot' },
  ];

  // Smooth animated zoom toward a target coordinate
  function animateTo(target, durationMs = 2000) {
    animating = true;
    const startCxHi = cx_hi, startCxLo = cx_lo;
    const startCyHi = cy_hi, startCyLo = cy_lo;
    const startZoom = zoom;
    const logZ0 = Math.log(startZoom);
    const logZ1 = Math.log(target.zoom);
    const ts = performance.now();

    function step(now) {
      const t = Math.min((now - ts) / durationMs, 1);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      zoom  = Math.exp(logZ0 + (logZ1 - logZ0) * e);
      cx_hi = startCxHi + (target.cx - startCxHi) * e;
      cx_lo = startCxLo * (1 - e);
      cy_hi = startCyHi + (target.cy - startCyHi) * e;
      cy_lo = startCyLo * (1 - e);

      if (useGL) {
        // GL renders every frame — no CSS trickery needed
        checkMode();
        glr.render(cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter(), W, H, pal);
      } else {
        const s  = 3.5 / (zoom * Math.min(W, H));
        const rs = 3.5 / (Math.exp(logZ0 + (logZ1 - logZ0) * Math.max(0, e - 0.05)) * Math.min(W, H));
        vScale = rs / s;
        vTx    = W * 0.5 * (1 - vScale);
        vTy    = H * 0.5 * (1 - vScale);
        applyCSS();
        if (t > 0.1) scheduleRender(80, false);
      }

      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        animating = false;
        cx_hi = target.cx; cx_lo = 0;
        cy_hi = target.cy; cy_lo = 0;
        zoom  = target.zoom;
        checkMode();
        if (useGL) { glr.render(cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter(), W, H, pal); updateHUD(); }
        else scheduleRender(0);
      }
    }
    requestAnimationFrame(step);
  }

  // Number keys 1-6 → jump to preset; R → home
  window.addEventListener('keydown', e => {
    const n = parseInt(e.key);
    if (n >= 1 && n <= PRESETS.length) {
      const p = PRESETS[n - 1];
      hud.textContent = `→ ${p.label}`;
      animateTo(p);
    }
    if (e.key === 'r' || e.key === 'R') animateTo(PRESETS[0]);
  });

  // ── UI buttons ────────────────────────────────────────────────────────────
  document.getElementById('btn-home').addEventListener('click', () => {
    animateTo(PRESETS[0]);
  });

  document.querySelectorAll('.loc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = PRESETS[parseInt(btn.dataset.preset)];
      hud.textContent = `→ ${p.label}`;
      animateTo(p);
    });
  });

  document.getElementById('pal-select').addEventListener('change', e => {
    pal = parseInt(e.target.value);
    if (useGL) markGlDirty(); else scheduleRender(0);
  });

  // ── Help modal ────────────────────────────────────────────────────────────
  const modalOverlay = document.getElementById('modal-overlay');
  document.getElementById('btn-help').addEventListener('click', () => {
    modalOverlay.classList.remove('hidden');
  });
  document.getElementById('modal-close').addEventListener('click', () => {
    modalOverlay.classList.add('hidden');
  });
  modalOverlay.addEventListener('click', e => {
    if (e.target === modalOverlay) modalOverlay.classList.add('hidden');
  });

  // Start at home (full Mandelbrot view)
  if (useGL) markGlDirty(); else renderParallel();

  // ── Zoom (scroll wheel) ───────────────────────────────────────────────────
  // CSS transform only — zero compute, instant GPU response
  evt.addEventListener('wheel', e => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.10 : 1 / 1.10;

    // Update Mandelbrot coords with DD-compensated arithmetic
    const s     = 3.5 / (zoom * Math.min(W, H));
    const delta = (1 - 1 / f);
    const dcx   = (e.clientX - W * 0.5) * s * delta;
    const dcy   = (e.clientY - H * 0.5) * s * delta;
    [cx_hi, cx_lo] = ddAdd(cx_hi, cx_lo, dcx);
    [cy_hi, cy_lo] = ddAdd(cy_hi, cy_lo, dcy);
    zoom *= f;

    if (useGL) {
      checkMode();      // may switch to WASM if zoom crossed threshold
      markGlDirty();
    } else {
      // Accumulate CSS zoom pinned to cursor position
      vTx = e.clientX + f * (vTx - e.clientX);
      vTy = e.clientY + f * (vTy - e.clientY);
      vScale *= f;
      applyCSS();
      scheduleInteractionRender();
      checkMode(); // switch back to GL if zoom dropped below threshold
    }
  }, { passive: false });

  // ── Pan (mouse drag) ──────────────────────────────────────────────────────
  let drag = null;
  evt.addEventListener('mousedown', e => {
    drag = { x: e.clientX, y: e.clientY, tx: vTx, ty: vTy,
             cx_hi, cx_lo, cy_hi, cy_lo };
  });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    const s  = 3.5 / (zoom * Math.min(W, H));
    [cx_hi, cx_lo] = ddAdd(drag.cx_hi, drag.cx_lo, -dx * s);
    [cy_hi, cy_lo] = ddAdd(drag.cy_hi, drag.cy_lo, -dy * s);

    if (useGL) {
      markGlDirty();
    } else {
      vTx = drag.tx + dx;
      vTy = drag.ty + dy;
      applyCSS();
      scheduleInteractionRender();
      checkMode();
    }
  });
  window.addEventListener('mouseup', () => { drag = null; });

  // ── Touch ─────────────────────────────────────────────────────────────────
  let lastTouches = null;
  evt.addEventListener('touchstart', e => {
    e.preventDefault();
    lastTouches = e.touches;
  }, { passive: false });
  evt.addEventListener('touchmove', e => {
    e.preventDefault();
    const t = e.touches;
    const s = 3.5 / (zoom * Math.min(W, H));
    if (t.length === 1 && lastTouches?.length === 1) {
      const dx = t[0].clientX - lastTouches[0].clientX;
      const dy = t[0].clientY - lastTouches[0].clientY;
      [cx_hi, cx_lo] = ddAdd(cx_hi, cx_lo, -dx * s);
      [cy_hi, cy_lo] = ddAdd(cy_hi, cy_lo, -dy * s);
      if (!useGL) { vTx += dx; vTy += dy; applyCSS(); }
    } else if (t.length === 2 && lastTouches?.length === 2) {
      const d0 = Math.hypot(lastTouches[0].clientX - lastTouches[1].clientX,
                            lastTouches[0].clientY - lastTouches[1].clientY);
      const d1 = Math.hypot(t[0].clientX - t[1].clientX,
                            t[0].clientY - t[1].clientY);
      const f  = d1 / d0;
      zoom *= f;
      if (!useGL) {
        const mx = (t[0].clientX + t[1].clientX) / 2;
        const my = (t[0].clientY + t[1].clientY) / 2;
        vTx = mx + f * (vTx - mx); vTy = my + f * (vTy - my); vScale *= f;
        applyCSS();
      }
    }
    lastTouches = t;
    if (useGL) { checkMode(); markGlDirty(); }
    else scheduleInteractionRender();
  }, { passive: false });

  // ── Resize ────────────────────────────────────────────────────────────────
  window.addEventListener('resize', () => {
    W = canvas.width  = glCanvas.width  = window.innerWidth;
    H = canvas.height = glCanvas.height = window.innerHeight;
    if (glr) glr.resize(W, H);
    resetCSS();
    if (useGL) markGlDirty(); else renderParallel();
  });
}

run().catch(console.error);
