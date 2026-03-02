// WebGL2 Mandelbrot renderer.
// Uses double-double (DD) arithmetic emulated with f32 pairs in GLSL,
// giving ~14 decimal digits of precision — sufficient up to zoom ~5e10.

const VERT = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;

uniform vec2  u_cx;      // DD center x  (hi_f32, lo_f32)
uniform vec2  u_cy;      // DD center y  (hi_f32, lo_f32)
uniform vec2  u_scale;   // DD pixel->Mandelbrot scale  (hi_f32, lo_f32)
uniform vec2  u_res;     // canvas (W, H)
uniform int   u_maxIter;
uniform int   u_pal;

out vec4 fragColor;

// ── Double-double arithmetic (two f32 simulate one f64) ──────────────────
vec2 dd_add(vec2 a, vec2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    return vec2(s, (a.x - (s - v)) + (b.x - v) + a.y + b.y);
}
vec2 dd_sub(vec2 a, vec2 b) { return dd_add(a, vec2(-b.x, -b.y)); }
vec2 dd_mul(vec2 a, vec2 b) {
    float p = a.x * b.x;
    // fma gives exact error of the a.x*b.x product
    return vec2(p, fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x);
}

// ── Cosine palette — exact match to Rust lib.rs ──────────────────────────
vec3 cospal(float sn, float mi, int p) {
    if (sn >= mi) return vec3(0.0);
    float t    = sn / mi * 10.0;
    float edge = 1.0 - exp(-sn * 0.04);
    const float TAU = 6.2831853072;
    vec3 c;
    if (p == 0) {                                      // Nebula
        c = vec3(0.5  + 0.6  * cos(TAU * (t + 0.00)),
                 0.5  + 0.6  * cos(TAU * (t + 0.35)),
                 0.5  + 0.6  * cos(TAU * (t + 0.72)));
    } else if (p == 1) {                               // Fire
        c = vec3(0.75 + 0.25 * cos(TAU *  t),
                 0.30 + 0.35 * cos(TAU * (t + 0.10)),
                 0.05 + 0.05 * cos(TAU * (t + 0.50)));
    } else if (p == 2) {                               // Ice
        c = vec3(0.25 + 0.25 * cos(TAU * (t + 0.55)),
                 0.55 + 0.45 * cos(TAU * (t + 0.70)),
                 0.80 + 0.20 * cos(TAU *  t));
    } else if (p == 3) {                               // Gold
        c = vec3(0.80 + 0.20 * cos(TAU * (t + 0.05)),
                 0.55 + 0.35 * cos(TAU * (t + 0.12)),
                 0.10 + 0.10 * cos(TAU * (t + 0.45)));
    } else if (p == 4) {                               // Ocean
        c = vec3(0.15 + 0.15 * cos(TAU * (t + 0.50)),
                 0.45 + 0.40 * cos(TAU * (t + 0.65)),
                 0.75 + 0.25 * cos(TAU *  t));
    } else if (p == 5) {                               // Sunset
        c = vec3(0.60 + 0.40 * cos(TAU *  t),
                 0.20 + 0.30 * cos(TAU * (t + 0.25)),
                 0.50 + 0.40 * cos(TAU * (t + 0.58)));
    } else if (p == 6) {                               // Forest
        c = vec3(0.25 + 0.25 * cos(TAU * (t + 0.15)),
                 0.55 + 0.35 * cos(TAU *  t),
                 0.10 + 0.12 * cos(TAU * (t + 0.40)));
    } else {                                           // Psychedelic
        float tp = t * 3.0;
        c = vec3(0.5  + 0.5  * cos(TAU *  tp),
                 0.5  + 0.5  * cos(TAU * (tp + 0.33)),
                 0.5  + 0.5  * cos(TAU * (tp + 0.67)));
    }
    return c * edge;
}

// ── Main ─────────────────────────────────────────────────────────────────
void main() {
    // Pixel offset from canvas centre; flip Y (WebGL origin = bottom-left)
    float ox =  gl_FragCoord.x - u_res.x * 0.5;
    float oy = -(gl_FragCoord.y - u_res.y * 0.5);

    // c = centre + offset * scale  (all in DD precision)
    vec2 cx = dd_add(u_cx, dd_mul(vec2(ox, 0.0), u_scale));
    vec2 cy = dd_add(u_cy, dd_mul(vec2(oy, 0.0), u_scale));

    // Mandelbrot iteration (DD z for accuracy at deep zoom)
    vec2 zx = vec2(0.0), zy = vec2(0.0);
    float sn = float(u_maxIter);

    for (int i = 0; i < u_maxIter; i++) {
        vec2 zx2 = dd_mul(zx, zx);
        vec2 zy2 = dd_mul(zy, zy);
        float mag = zx2.x + zy2.x;          // hi part sufficient for escape test
        if (mag > 4.0) {
            sn = float(i) + 1.0 - log(log(mag) * 0.5 / log(2.0)) / log(2.0);
            break;
        }
        vec2 nzx = dd_add(dd_sub(zx2, zy2), cx);
        vec2 nzy = dd_add(dd_mul(dd_add(zx, zx), zy), cy);
        zx = nzx; zy = nzy;
    }

    fragColor = vec4(cospal(sn, float(u_maxIter), u_pal), 1.0);
}`;

export class GlRenderer {
    constructor(canvas) {
        const gl = canvas.getContext('webgl2', { antialias: false, depth: false, alpha: false });
        if (!gl) throw new Error('WebGL2 not supported');
        this.gl = gl;

        const vs = this._shader(gl.VERTEX_SHADER,   VERT);
        const fs = this._shader(gl.FRAGMENT_SHADER, FRAG);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
            throw new Error(gl.getProgramInfoLog(prog));
        gl.useProgram(prog);

        // Full-screen quad (2 triangles)
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER,
            new Float32Array([-1,-1, 1,-1, -1,1,  1,-1, 1,1, -1,1]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, 'a_pos');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

        const U = n => gl.getUniformLocation(prog, n);
        this.u = { cx: U('u_cx'), cy: U('u_cy'), scale: U('u_scale'),
                   res: U('u_res'), maxIter: U('u_maxIter'), pal: U('u_pal') };
    }

    _shader(type, src) {
        const s = this.gl.createShader(type);
        this.gl.shaderSource(s, src);
        this.gl.compileShader(s);
        if (!this.gl.getShaderParameter(s, this.gl.COMPILE_STATUS))
            throw new Error(this.gl.getShaderInfoLog(s));
        return s;
    }

    resize(w, h) { this.gl.viewport(0, 0, w, h); }

    // Convert a JS f64 DD pair (hi+lo) into two f32 values for GLSL uniforms.
    // hi_f32 + lo_f32  ≈  hi_f64 + lo_f64  with ~14 decimal digits.
    static ddToF32(hi, lo) {
        const h = Math.fround(hi);
        return [h, Math.fround(hi - h + lo)];
    }

    render(cx_hi, cx_lo, cy_hi, cy_lo, zoom, maxIter, W, H, pal) {
        const { gl, u } = this;
        const scale  = 3.5 / (zoom * Math.min(W, H));
        const sh     = Math.fround(scale);
        const [cxh, cxl] = GlRenderer.ddToF32(cx_hi, cx_lo);
        const [cyh, cyl] = GlRenderer.ddToF32(cy_hi, cy_lo);
        gl.uniform2f(u.cx,      cxh, cxl);
        gl.uniform2f(u.cy,      cyh, cyl);
        gl.uniform2f(u.scale,   sh,  Math.fround(scale - sh));
        gl.uniform2f(u.res,     W,   H);
        gl.uniform1i(u.maxIter, maxIter);
        gl.uniform1i(u.pal,     pal);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}
