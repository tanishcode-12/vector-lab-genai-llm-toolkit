/*
 * Signature hero visual: a small animated "embedding space" — points
 * drift slowly, and the nearest few pairs are connected with faint
 * lines, echoing the CBOW / Skip-gram vector geometry explained on
 * this page. Purely decorative, no data dependency.
 */
(function () {
  const canvas = document.getElementById("vectorField");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  const WORDS = [
    "small", "big", "team", "sun", "mountain", "beautiful",
    "grand", "tiny", "success", "garden", "ocean", "storm",
    "smile", "record", "flower", "city", "child", "dream",
  ];

  let width, height, points;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height || 440;
    canvas.width = width * DPR;
    canvas.height = height * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function seedPoints() {
    points = WORDS.map((word, i) => {
      const angle = (i / WORDS.length) * Math.PI * 2;
      const radius = 0.28 + 0.16 * Math.sin(i * 1.7);
      return {
        word,
        x: 0.5 + radius * Math.cos(angle),
        y: 0.5 + radius * Math.sin(angle),
        vx: (Math.random() - 0.5) * 0.00025,
        vy: (Math.random() - 0.5) * 0.00025,
        phase: Math.random() * Math.PI * 2,
      };
    });
  }

  function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function step(t) {
    ctx.clearRect(0, 0, width, height);

    // gentle drift
    points.forEach((p) => {
      p.x += p.vx + Math.sin(t / 4000 + p.phase) * 0.00002;
      p.y += p.vy + Math.cos(t / 4000 + p.phase) * 0.00002;
      p.x = Math.min(0.94, Math.max(0.06, p.x));
      p.y = Math.min(0.94, Math.max(0.06, p.y));
    });

    // connect nearest neighbours (k=2) with faint lines
    ctx.lineWidth = 1;
    points.forEach((p) => {
      const neighbours = points
        .filter((q) => q !== p)
        .map((q) => ({ q, d: distance(p, q) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 2);

      neighbours.forEach(({ q, d }) => {
        const alpha = Math.max(0, 0.22 - d * 0.4);
        ctx.strokeStyle = `rgba(110, 231, 216, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(p.x * width, p.y * height);
        ctx.lineTo(q.x * width, q.y * height);
        ctx.stroke();
      });
    });

    // draw points + labels
    points.forEach((p) => {
      const px = p.x * width;
      const py = p.y * height;

      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#6ee7d8";
      ctx.fill();

      ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(139, 147, 167, 0.85)";
      ctx.fillText(p.word, px + 7, py + 3);
    });

    requestAnimationFrame(step);
  }

  resize();
  seedPoints();
  window.addEventListener("resize", () => {
    resize();
  });

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!prefersReducedMotion) {
    requestAnimationFrame(step);
  } else {
    step(0);
  }
})();
