use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// Exposes raw WASM memory to JS for zero-copy pixel reads.
#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub struct Renderer {
    width: u32,
    height: u32,
    buf: Vec<u8>, // RGBA, row-major
}

#[wasm_bindgen]
impl Renderer {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buf: vec![0u8; (width * height * 4) as usize],
        }
    }

    /// Pointer into WASM heap — JS reads this via memory.buffer.
    pub fn buf_ptr(&self) -> u32 {
        self.buf.as_ptr() as u32
    }

    pub fn render(&mut self, cx: f64, cy: f64, zoom: f64, max_iter: u32, pal: u32) {
        let w = self.width as f64;
        let h = self.height as f64;
        let scale = 3.5 / (zoom * w.min(h));

        for py in 0..self.height {
            for px in 0..self.width {
                let x = (px as f64 - w * 0.5) * scale + cx;
                let y = (py as f64 - h * 0.5) * scale + cy;

                let s = escape(x, y, max_iter);
                let [r, g, b] = palette(s, max_iter, pal);

                let i = ((py * self.width + px) * 4) as usize;
                self.buf[i]     = r;
                self.buf[i + 1] = g;
                self.buf[i + 2] = b;
                self.buf[i + 3] = 255;
            }
        }
    }
}

// ── SIMD escape: 2 pixels per iteration ──────────────────────────────────────
// Processes two horizontally adjacent pixels simultaneously using f64x2 SIMD.
// Both pixels share the same y coordinate, so cy is scalar — only cx differs.
// We run until both pixels have escaped (or hit max_iter), then apply smooth
// coloring to each lane independently.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn escape_x2(cx0: f64, cy_v: f64, cx1: f64, max_iter: u32) -> (f64, f64) {
    let cx   = f64x2(cx0, cx1);
    let cy   = f64x2_splat(cy_v);
    let mut zx = f64x2_splat(0.0);
    let mut zy = f64x2_splat(0.0);
    let mut n   = [max_iter; 2];
    let mut mag = [0.0f64;   2];

    for iter in 0..max_iter {
        let zx2  = f64x2_mul(zx, zx);
        let zy2  = f64x2_mul(zy, zy);
        let mag2 = f64x2_add(zx2, zy2);

        if n[0] == max_iter {
            let m = f64x2_extract_lane::<0>(mag2);
            if m > 4.0 { n[0] = iter; mag[0] = m; }
        }
        if n[1] == max_iter {
            let m = f64x2_extract_lane::<1>(mag2);
            if m > 4.0 { n[1] = iter; mag[1] = m; }
        }
        if n[0] < max_iter && n[1] < max_iter { break; }

        let new_zy = f64x2_add(f64x2_mul(f64x2_add(zx, zx), zy), cy);
        let new_zx = f64x2_add(f64x2_sub(zx2, zy2), cx);
        zx = new_zx;
        zy = new_zy;
    }

    let smooth = |ni: u32, m: f64| -> f64 {
        if ni == max_iter || m <= 0.0 { return max_iter as f64; }
        let log_zn = m.ln() * 0.5;
        let nu = (log_zn / core::f64::consts::LN_2).ln() / core::f64::consts::LN_2;
        ni as f64 + 1.0 - nu
    };
    (smooth(n[0], mag[0]), smooth(n[1], mag[1]))
}

/// Renders a horizontal strip of the full canvas — called by Web Workers in parallel.
/// Returns an RGBA Vec<u8> of size canvas_w × (y_end - y_start) × 4.
/// Uses SIMD f64x2 on wasm32 to process 2 pixels per iteration.
#[wasm_bindgen]
pub fn render_strip(
    cx: f64, cy: f64, zoom: f64, max_iter: u32,
    canvas_w: u32, canvas_h: u32,
    y_start: u32, y_end: u32, pal: u32,
) -> Vec<u8> {
    let strip_h = y_end - y_start;
    let mut buf = vec![0u8; (canvas_w * strip_h * 4) as usize];

    let w = canvas_w as f64;
    let h = canvas_h as f64;
    let scale = 3.5 / (zoom * w.min(h));

    for py in 0..strip_h {
        let global_y = (py + y_start) as f64;
        let y_c = (global_y - h * 0.5) * scale + cy;

        #[cfg(target_arch = "wasm32")]
        {
            let mut px = 0u32;
            while px + 1 < canvas_w {
                let x0 = (px as f64       - w * 0.5) * scale + cx;
                let x1 = ((px + 1) as f64 - w * 0.5) * scale + cx;
                let (s0, s1) = unsafe { escape_x2(x0, y_c, x1, max_iter) };

                let [r0, g0, b0] = palette(s0, max_iter, pal);
                let [r1, g1, b1] = palette(s1, max_iter, pal);

                let i0 = ((py * canvas_w + px) * 4) as usize;
                buf[i0]     = r0; buf[i0 + 1] = g0; buf[i0 + 2] = b0; buf[i0 + 3] = 255;
                let i1 = i0 + 4;
                buf[i1]     = r1; buf[i1 + 1] = g1; buf[i1 + 2] = b1; buf[i1 + 3] = 255;
                px += 2;
            }
            // Handle odd last column
            if px < canvas_w {
                let x = (px as f64 - w * 0.5) * scale + cx;
                let s = escape(x, y_c, max_iter);
                let [r, g, b] = palette(s, max_iter, pal);
                let i = ((py * canvas_w + px) * 4) as usize;
                buf[i] = r; buf[i+1] = g; buf[i+2] = b; buf[i+3] = 255;
            }
        }

        // Non-wasm32 scalar fallback (for native tests/dev)
        #[cfg(not(target_arch = "wasm32"))]
        for px in 0..canvas_w {
            let x = (px as f64 - w * 0.5) * scale + cx;
            let s = escape(x, y_c, max_iter);
            let [r, g, b] = palette(s, max_iter, pal);
            let i = ((py * canvas_w + px) * 4) as usize;
            buf[i] = r; buf[i+1] = g; buf[i+2] = b; buf[i+3] = 255;
        }
    }

    buf
}

// ── Double-double arithmetic ──────────────────────────────────────────────
// Represents a number as (hi + lo) with ~32 decimal digits of precision.
// Extends usable zoom from ~1e13 (f64 limit) to ~1e26.

type DD = (f64, f64);

#[inline] fn dd(x: f64) -> DD { (x, 0.0) }

/// Exact sum of two f64s as a DD.
#[inline]
fn two_sum(a: f64, b: f64) -> DD {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// Exact product of two f64s as a DD (uses FMA).
#[inline]
fn two_prod(a: f64, b: f64) -> DD {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

#[inline]
fn dd_add(a: DD, b: DD) -> DD {
    let (s, e) = two_sum(a.0, b.0);
    two_sum(s, e + a.1 + b.1)
}

#[inline]
fn dd_sub(a: DD, b: DD) -> DD { dd_add(a, (-b.0, -b.1)) }

#[inline]
fn dd_mul(a: DD, b: DD) -> DD {
    let (p, e) = two_prod(a.0, b.0);
    two_sum(p, e + a.0 * b.1 + a.1 * b.0)
}

/// Mandelbrot escape using full DD arithmetic — for zoom > 1e11.
fn escape_dd(cx: DD, cy: DD, max_iter: u32) -> f64 {
    let mut x = dd(0.0);
    let mut y = dd(0.0);
    let mut n = 0u32;

    while n < max_iter {
        let x2 = dd_mul(x, x);
        let y2 = dd_mul(y, y);
        if x2.0 + y2.0 > 4.0 { break; }
        let ny = dd_add(dd_mul((2.0 * x.0, 2.0 * x.1), y), cy);
        let nx = dd_add(dd_sub(x2, y2), cx);
        x = nx; y = ny;
        n += 1;
    }

    if n == max_iter { return max_iter as f64; }

    let mag2 = x.0 * x.0 + y.0 * y.0;
    let log_zn = mag2.ln() * 0.5;
    let nu = (log_zn / std::f64::consts::LN_2).ln() / std::f64::consts::LN_2;
    n as f64 + 1.0 - nu
}

/// High-precision strip renderer — used automatically when zoom > 1e11.
/// Takes center as (hi, lo) double-double pairs for ~1e26 zoom depth.
#[wasm_bindgen]
pub fn render_strip_dd(
    cx_hi: f64, cx_lo: f64,
    cy_hi: f64, cy_lo: f64,
    zoom: f64, max_iter: u32,
    canvas_w: u32, canvas_h: u32,
    y_start: u32, y_end: u32, pal: u32,
) -> Vec<u8> {
    let strip_h = y_end - y_start;
    let mut buf = vec![0u8; (canvas_w * strip_h * 4) as usize];

    let w = canvas_w as f64;
    let h = canvas_h as f64;
    let scale = 3.5 / (zoom * w.min(h));
    let cx = (cx_hi, cx_lo);
    let cy = (cy_hi, cy_lo);

    for py in 0..strip_h {
        let gy = (py + y_start) as f64;
        for px in 0..canvas_w {
            let dx = (px as f64 - w * 0.5) * scale;
            let dy = (gy           - h * 0.5) * scale;
            // Pixel center = DD center + f64 offset (offset is small, no precision loss)
            let c_x = dd_add(cx, dd(dx));
            let c_y = dd_add(cy, dd(dy));

            let s = escape_dd(c_x, c_y, max_iter);
            let [r, g, b] = palette(s, max_iter, pal);

            let i = ((py * canvas_w + px) * 4) as usize;
            buf[i] = r; buf[i+1] = g; buf[i+2] = b; buf[i+3] = 255;
        }
    }
    buf
}

/// Returns smooth (continuous) escape iteration count.
fn escape(cx: f64, cy: f64, max_iter: u32) -> f64 {
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let mut n = 0u32;

    while n < max_iter {
        let x2 = x * x;
        let y2 = y * y;
        if x2 + y2 > 4.0 {
            break;
        }
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        n += 1;
    }

    if n == max_iter {
        return max_iter as f64;
    }

    // Bernard smooth coloring formula
    let mag2 = x * x + y * y;
    let log_zn = mag2.ln() * 0.5;
    let nu = (log_zn / std::f64::consts::LN_2).ln() / std::f64::consts::LN_2;
    n as f64 + 1.0 - nu
}

/// Maps smooth iteration count to an RGB color.
/// pal selects one of three cosine palettes (Inigo Quilez formula):
///   color(t) = a + b * cos(TAU * (c*t + d))
fn palette(s: f64, max_iter: u32, pal: u32) -> [u8; 3] {
    if s >= max_iter as f64 {
        return [0, 0, 0];
    }

    // 10 color cycles — more variation, sharper detail differentiation.
    let t = s / max_iter as f64 * 10.0;

    // Gentle boundary darkening so deep-boundary pixels glow without going muddy.
    let edge = 1.0 - (-s * 0.04).exp();

    let (r, g, b) = match pal {
        // 0 · Nebula — vivid purple → teal → gold cycle (higher amplitude = more contrast)
        0 => (
            cos_channel(t, 0.5, 0.6, 1.0, 0.00),
            cos_channel(t, 0.5, 0.6, 1.0, 0.35),
            cos_channel(t, 0.5, 0.6, 1.0, 0.72),
        ),
        // 1 · Fire — deep red → orange → bright yellow
        1 => (
            cos_channel(t, 0.75, 0.25, 1.0, 0.00),
            cos_channel(t, 0.30, 0.35, 1.0, 0.10),
            cos_channel(t, 0.05, 0.05, 1.0, 0.50),
        ),
        // 2 · Ice — navy → cyan → white
        _ => (
            cos_channel(t, 0.25, 0.25, 1.0, 0.55),
            cos_channel(t, 0.55, 0.45, 1.0, 0.70),
            cos_channel(t, 0.80, 0.20, 1.0, 0.00),
        ),
    };

    [
        (r * edge * 255.0) as u8,
        (g * edge * 255.0) as u8,
        (b * edge * 255.0) as u8,
    ]
}

#[inline]
fn cos_channel(t: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    a + b * (std::f64::consts::TAU * (c * t + d)).cos()
}

// ── Perturbation theory ───────────────────────────────────────────────────────
// Compute one high-precision reference orbit at the center using DD arithmetic,
// then render every pixel as a cheap f64 perturbation δ around that orbit.
//
// Key identity: if Z_n is the reference orbit and z_n = Z_n + δ_n is the pixel
// orbit, then:
//   δ_{n+1} = 2·Z_n·δ_n + δ_n² + ε   (ε = pixel offset from center, tiny)
//
// δ stays small in f64 even at extreme zoom because it's measured relative to
// the center, not in absolute coordinates. This gives ~8× speedup over full DD
// per-pixel arithmetic.

/// Computes the DD reference orbit at (cx, cy) and returns it as flattened
/// f64 pairs [z0x, z0y, z1x, z1y, …] up to escape or max_iter.
fn compute_reference_orbit_internal(
    cx_hi: f64, cx_lo: f64,
    cy_hi: f64, cy_lo: f64,
    max_iter: u32,
) -> Vec<f64> {
    let cx = (cx_hi, cx_lo);
    let cy = (cy_hi, cy_lo);
    let mut zx = dd(0.0);
    let mut zy = dd(0.0);
    let mut orbit = Vec::with_capacity((max_iter as usize) * 2);

    for _ in 0..max_iter {
        // Best single-f64 approximation of the DD value (hi + lo rounded)
        orbit.push(zx.0 + zx.1);
        orbit.push(zy.0 + zy.1);

        let zx2 = dd_mul(zx, zx);
        let zy2 = dd_mul(zy, zy);
        if zx2.0 + zy2.0 > 4.0 { break; }

        let new_zy = dd_add(dd_mul((2.0 * zx.0, 2.0 * zx.1), zy), cy);
        let new_zx = dd_add(dd_sub(zx2, zy2), cx);
        zx = new_zx;
        zy = new_zy;
    }
    orbit
}

/// Escape test using perturbation arithmetic.
/// orbit: flattened [z0x, z0y, z1x, z1y, …] reference orbit.
/// eps_x/eps_y: pixel offset from reference center (small, f64 is fine).
/// cx_*/cy_*: reference center in DD — used for accurate fallback.
///
/// Glitch scenario: if the reference orbit escaped before max_iter, pixels
/// that need more iterations than orbit.len() can't be tracked. We detect
/// this and fall back to full escape_dd for those pixels.
fn escape_perturb(
    orbit: &[f64],
    eps_x: f64, eps_y: f64,
    max_iter: u32,
    cx_hi: f64, cx_lo: f64,
    cy_hi: f64, cy_lo: f64,
) -> f64 {
    let ref_len = orbit.len() / 2;
    let mut dx = 0.0f64; // δ_0 = 0
    let mut dy = 0.0f64;

    for n in 0..ref_len {
        let zx = orbit[n * 2];
        let zy = orbit[n * 2 + 1];

        // Full orbit value z_n = Z_n + δ_n
        let full_x = zx + dx;
        let full_y = zy + dy;
        let mag2   = full_x * full_x + full_y * full_y;

        if mag2 > 4.0 {
            let log_zn = mag2.ln() * 0.5;
            let nu = (log_zn / std::f64::consts::LN_2).ln() / std::f64::consts::LN_2;
            return n as f64 + 1.0 - nu;
        }

        // δ_{n+1} = 2·Z_n·δ_n + δ_n² + ε
        let dx2  = dx * dx;
        let dy2  = dy * dy;
        let dxdy = dx * dy;
        let new_dx = 2.0 * (zx * dx - zy * dy) + dx2 - dy2 + eps_x;
        let new_dy = 2.0 * (zx * dy + zy * dx) + 2.0 * dxdy + eps_y;
        dx = new_dx;
        dy = new_dy;
    }

    // Orbit exhausted. Two cases:
    // 1. ref_len == max_iter: reference is interior → this pixel is interior too.
    // 2. ref_len < max_iter: reference escaped early but this pixel hasn't yet.
    //    Perturbation can't continue — fall back to accurate DD for this pixel.
    if ref_len < max_iter as usize {
        let pixel_cx = dd_add((cx_hi, cx_lo), dd(eps_x));
        let pixel_cy = dd_add((cy_hi, cy_lo), dd(eps_y));
        return escape_dd(pixel_cx, pixel_cy, max_iter);
    }

    max_iter as f64
}

/// Perturbation-theory strip renderer — drop-in replacement for render_strip_dd.
/// Computes one DD reference orbit (once per strip) then uses f64 perturbations
/// for every pixel. Roughly 8× faster than render_strip_dd at deep zoom.
#[wasm_bindgen]
pub fn render_strip_perturb(
    cx_hi: f64, cx_lo: f64,
    cy_hi: f64, cy_lo: f64,
    zoom: f64, max_iter: u32,
    canvas_w: u32, canvas_h: u32,
    y_start: u32, y_end: u32, pal: u32,
) -> Vec<u8> {
    let orbit   = compute_reference_orbit_internal(cx_hi, cx_lo, cy_hi, cy_lo, max_iter);
    let strip_h = y_end - y_start;
    let mut buf = vec![0u8; (canvas_w * strip_h * 4) as usize];

    let w     = canvas_w as f64;
    let h     = canvas_h as f64;
    let scale = 3.5 / (zoom * w.min(h));

    for py in 0..strip_h {
        let gy = (py + y_start) as f64;
        for px in 0..canvas_w {
            let eps_x = (px as f64 - w * 0.5) * scale;
            let eps_y = (gy          - h * 0.5) * scale;

            let s = escape_perturb(&orbit, eps_x, eps_y, max_iter, cx_hi, cx_lo, cy_hi, cy_lo);
            let [r, g, b] = palette(s, max_iter, pal);

            let i = ((py * canvas_w + px) * 4) as usize;
            buf[i] = r; buf[i+1] = g; buf[i+2] = b; buf[i+3] = 255;
        }
    }
    buf
}
