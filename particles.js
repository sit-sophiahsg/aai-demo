// =============================================================================
//  particles.js: a fixed-size, allocation-free particle system
// =============================================================================
//
//  WHY A POOL?
//  Creating a new object per particle (hundreds per second) makes garbage
//  the browser must collect, which shows up as stutter. Instead, all particle
//  data lives in typed arrays allocated ONCE. Alive particles are packed at
//  indices 0..count-1. To remove particle i, we copy the last alive particle
//  into slot i and shrink `count` ("swap-remove"), so there are never holes.
//
//  WHY SPRITES?
//  A soft glow needs a radial gradient, and building a gradient per particle
//  per frame is slow. We pre-render one glow image per color to a small
//  offscreen canvas, then just stamp it with drawImage().
//
//  GLOW
//  globalCompositeOperation = "lighter" ADDS colors together, so overlapping
//  particles get brighter, like real light.
// =============================================================================

// Particles are only ~5-25 px on screen, so a 32 px sprite is plenty and
// is cheaper for the GPU to sample than a bigger one.
const SPRITE_SIZE = 32;

export class ParticleSystem {
  constructor(capacity = 300) {
    this.capacity = capacity;
    this.count = 0;
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
    this.vx = new Float32Array(capacity);
    this.vy = new Float32Array(capacity);
    this.life = new Float32Array(capacity);     // seconds left
    this.maxLife = new Float32Array(capacity);  // starting lifetime
    this.size = new Float32Array(capacity);     // starting radius, canvas px
    this.color = new Array(capacity).fill("#ffffff"); // references to cached strings
    this.sprites = new Map();                   // color -> glow canvas

    this.drag = 2.0;   // velocity decays by e^(-drag * t): particles slow down
    this.lift = -25;   // px/s² upward drift, like embers rising
  }

  // Add one particle. Returns false if the pool is full (the cap keeps FPS safe).
  spawn(x, y, vx, vy, size, life, color) {
    if (this.count >= this.capacity) return false;
    const i = this.count++;
    this.x[i] = x;
    this.y[i] = y;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.size[i] = size;
    this.life[i] = life;
    this.maxLife[i] = life;
    this.color[i] = color;
    return true;
  }

  // A radial explosion of n particles (used for bubble pops later).
  burst(x, y, color, n, speed, size, life = 0.6) {
    for (let k = 0; k < n; k++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.4 + Math.random() * 0.6);
      if (!this.spawn(x, y, Math.cos(a) * s, Math.sin(a) * s,
        size * (0.6 + Math.random() * 0.6), life * (0.7 + Math.random() * 0.6), color)) break;
    }
  }

  update(dt) {
    const damp = Math.exp(-this.drag * dt);
    let i = 0;
    while (i < this.count) {
      this.life[i] -= dt;
      if (this.life[i] <= 0) {
        // swap-remove: move the last alive particle into this slot
        const last = --this.count;
        this.x[i] = this.x[last];
        this.y[i] = this.y[last];
        this.vx[i] = this.vx[last];
        this.vy[i] = this.vy[last];
        this.size[i] = this.size[last];
        this.life[i] = this.life[last];
        this.maxLife[i] = this.maxLife[last];
        this.color[i] = this.color[last];
        continue; // re-check slot i, it now holds a different particle
      }
      this.vx[i] *= damp;
      this.vy[i] = this.vy[i] * damp + this.lift * dt;
      this.x[i] += this.vx[i] * dt;
      this.y[i] += this.vy[i] * dt;
      i++;
    }
  }

  render(ctx) {
    if (this.count === 0) return;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < this.count; i++) {
      const t = this.life[i] / this.maxLife[i];    // 1 = just born, 0 = dead
      const r = this.size[i] * (0.25 + 0.75 * t) * 2; // shrink; *2 for the halo
      ctx.globalAlpha = t;                         // fade
      ctx.drawImage(this.sprite(this.color[i]), this.x[i] - r, this.y[i] - r, r * 2, r * 2);
    }
    ctx.restore();
  }

  clear() {
    this.count = 0;
  }

  // Pre-rendered glow for a color: hot white core -> color -> transparent.
  sprite(color) {
    let c = this.sprites.get(color);
    if (!c) {
      c = document.createElement("canvas");
      c.width = c.height = SPRITE_SIZE;
      const g = c.getContext("2d");
      const h = SPRITE_SIZE / 2;
      const grad = g.createRadialGradient(h, h, 0, h, h, h);
      grad.addColorStop(0, "rgba(255,255,255,1)");
      grad.addColorStop(0.2, color);
      grad.addColorStop(0.45, color + "88"); // hex colors only: adds alpha
      grad.addColorStop(1, color + "00");
      g.fillStyle = grad;
      g.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
      this.sprites.set(color, c);
    }
    return c;
  }
}
