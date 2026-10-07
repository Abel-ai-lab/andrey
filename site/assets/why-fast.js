// "Why Andrey is fast": plays each scene with Motion (motion.dev, vendored as motion.js).
// Enhancement only: without this script every scene still shows its lanes and final counts.
// Each scene is one Motion sequence; its counters follow a clock that pauses with it, because a
// sequence of springs does not report its own time.
(() => {
  const Motion = window.Motion;
  if (!Motion) return;
  const { animate, stagger, inView } = Motion;
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const SETTLE = { type: "spring", bounce: 0.18, visualDuration: 0.5 };
  const QUICK = { type: "spring", bounce: 0.14, visualDuration: 0.32 };
  const LINE = 38; // the canvas's working line, in pixels from its top
  const WIDTH = 560; // every lane draws on a canvas this wide, scaled to its track
  const HEIGHT = 84;
  const EDGE =
    '<svg viewBox="0 0 26 12" aria-hidden="true"><line class="wf-line" x1="5" y1="6" x2="21" ' +
    'y2="6"/><circle cx="5" cy="6" r="3.6"/><circle cx="21" cy="6" r="3.6"/></svg>';
  // A correlation-like pattern for the matrices: a strong diagonal, symmetric, a few strong pairs.
  const SQUARE = [[5, 3, 1, 2], [3, 5, 2, 1], [1, 2, 5, 4], [2, 1, 4, 5]];
  const level = (i, j, rows) => (rows === 4 && j < 4 ? SQUARE[i][j] : 1 + ((i * 7 + j * 3 + ((i + j) % 3) * 2) % 5));

  const make = (tag, cls, parent) => {
    const node = document.createElement(tag);
    node.className = cls;
    parent.append(node);
    return node;
  };

  // ---- building a scene ---------------------------------------------------------------------
  // Both lanes share one layout, computed from the tracks' common width, so a station or a pile
  // sits at the same place in each lane.
  function lanes(figure) {
    return [...figure.querySelectorAll(".wf-lane")].map((row) => {
      const holder = row.querySelector(".wf-track");
      holder.replaceChildren();
      const track = make("div", "wf-canvas", holder);
      return {
        who: row.classList.contains("andrey") ? "andrey" : "others",
        holder,
        track,
        num: row.querySelector(".wf-count b"),
        unit: row.querySelector(".wf-count small"),
        badge: row.querySelector(".wf-done"),
      };
    });
  }

  // Scale each lane's canvas to its track, centered when the track is wider.
  function fit(figure) {
    for (const holder of figure.querySelectorAll(".wf-track")) {
      const canvas = holder.querySelector(".wf-canvas");
      if (!canvas) continue;
      const scale = Math.min(1.12, holder.clientWidth / WIDTH);
      const left = (holder.clientWidth - WIDTH * scale) / 2;
      canvas.style.transform = `translateX(${left}px) scale(${scale})`;
      holder.style.height = `${HEIGHT * scale}px`;
    }
  }
  const resizing = new ResizeObserver((entries) => entries.forEach((entry) => fit(entry.target)));

  // Two rows of slots from x0 to x1: where edges wait and where they land.
  const pile = (x0, x1, n) => {
    const cols = Math.ceil(n / 2);
    const step = Math.min(34, (x1 - x0 - 26) / Math.max(1, cols - 1));
    return (i) => ({ x: x0 + Math.floor(i / 2) * step, y: i % 2 ? 11 : -11 });
  };

  function station(track, x, w, h, tag, gate = false) {
    const box = make("div", "wf-station" + (gate ? " gate" : ""), track);
    box.style.cssText = `left:${x}px;top:${LINE - h / 2}px;width:${w}px;height:${h}px`;
    const glow = make("div", "wf-glow", box);
    const label = make("div", "wf-tag", track);
    label.textContent = tag;
    label.style.left = `${x + w / 2}px`;
    return { box, glow, x, w };
  }

  function edges(track, n) {
    return Array.from({ length: n }, () => {
      const edge = make("div", "wf-edge", track);
      edge.innerHTML = EDGE;
      return edge;
    });
  }

  function matrix(track, x, rows, cols, size, gap) {
    const top = LINE - (rows * (size + gap) - gap) / 2;
    const cells = [];
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = level(i, j, rows);
        const cell = make("div", `wf-cell r${value}`, track);
        cell.style.cssText =
          `left:${x + j * (size + gap)}px;top:${top + i * (size + gap)}px;` +
          `width:${size}px;height:${size}px;--v:${(0.18 + value * 0.16).toFixed(2)}`;
        cells.push(cell);
      }
    }
    return cells;
  }

  // A sequence under construction: every move states where it starts, so the sequence holds each
  // element in place until its turn.
  function score() {
    const seq = [];
    const events = [];
    const where = new Map();
    return {
      seq,
      events,
      place(node, x, y) {
        where.set(node, { x, y });
        node.style.transform = `translateX(${x}px) translateY(${y}px)`;
      },
      move(node, x, y, at, feel = SETTLE, arc = 0) {
        const from = where.get(node);
        const lift = Math.min(from.y, y) - arc;
        const ys = arc ? [from.y, lift, y] : [from.y, y];
        const yFeel = arc ? { duration: feel.visualDuration * 1.15, ease: "easeInOut" } : feel;
        seq.push([node, { x: [from.x, x], y: ys }, { at, x: feel, y: yFeel }]);
        where.set(node, { x, y });
      },
      glow(target, at) {
        seq.push([target.glow, { opacity: [0, 0.55, 0], scale: [0.94, 1, 1.04] }, { duration: 0.6, at, ease: "easeOut" }]);
        seq.push([target.box, { scale: [1, 1.045, 1] }, { duration: 0.34, at, ease: "easeOut" }]);
      },
      count(at, lane, value) { events.push({ at, lane, value }); },
      done(at, lane) { events.push({ at, lane, done: true }); },
    };
  }

  // ---- the four scenes ----------------------------------------------------------------------
  // Each builds both lanes into one sequence and returns its length in seconds.
  const SCENES = {
    parallel(L, S, W, data) {
      const n = data.n;
      const queue = pile(0, W * 0.3, n);
      const out = pile(W * 0.6, W, n);
      const parts = L.map((lane) => {
        const st = station(lane.track, Math.round(W * 0.38), 110, 50, "compute");
        const es = edges(lane.track, n);
        es.forEach((e, i) => S.place(e, queue(i).x, queue(i).y));
        return { st, es };
      });
      const [a, o] = parts;
      a.es.forEach((e, i) => S.move(e, a.st.x + 7 + (i % 4) * 24, i < 4 ? -10 : 10, 0.6 + i * 0.035, SETTLE, 14));
      S.glow(a.st, 1.1);
      S.count(1.12, 0, 1);
      a.es.forEach((e, i) => S.move(e, out(i).x, out(i).y, 1.35 + i * 0.035, SETTLE, 10));
      S.done(1.95, 0);
      o.es.forEach((e, i) => {
        const t = 0.6 + i * 0.46;
        S.move(e, o.st.x + 42, 0, t, QUICK, 10);
        S.glow(o.st, t + 0.24);
        S.count(t + 0.26, 1, i + 1);
        S.move(e, out(i).x, out(i).y, t + 0.3, QUICK, 8);
      });
      S.done(0.6 + (n - 1) * 0.46 + 0.62, 1);
      return 0.6 + (n - 1) * 0.46 + 0.9;
    },

    reuse(L, S, W, data) {
      const n = data.n;
      const queue = pile(50, W * 0.32, n);
      const out = pile(W * 0.6, W, n);
      const pairs = [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3]];
      const parts = L.map((lane) => {
        const bars = Array.from({ length: 5 }, (_, k) => {
          const bar = make("div", "wf-bar", lane.track);
          bar.style.top = `${LINE - 16 + k * 7}px`;
          return bar;
        });
        const tag = make("div", "wf-tag", lane.track);
        tag.textContent = "data";
        tag.style.left = "15px";
        const st = station(lane.track, Math.round(W * 0.42), 56, 56, "correlation matrix");
        const cells = matrix(lane.track, st.x + 8.5, 4, 4, 8.5, 2.5);
        const bits = Array.from({ length: 3 }, () => make("div", "wf-bit", lane.track));
        const es = edges(lane.track, n);
        es.forEach((e, i) => S.place(e, queue(i).x, queue(i).y));
        cells.forEach((c) => { c.style.opacity = 0; });
        return { bars, st, cells, bits, es };
      });
      const compute = (p, at) => {
        S.seq.push([p.bars, { opacity: [1, 0.35, 1] }, { duration: 0.5, delay: stagger(0.05), at }]);
        S.seq.push([p.bits, { x: [34, p.st.x - 10], opacity: [0, 1, 0] }, { duration: 0.45, delay: stagger(0.07), at: at + 0.05, ease: "easeInOut" }]);
        S.seq.push([p.cells, { scale: [0.3, 1], opacity: [0, 1] }, { ...QUICK, delay: stagger(0.018), at: at + 0.28 }]);
      };
      const read = (p, e, pair, at, slot) => {
        const [i, j] = pair;
        const hit = [i * 4 + i, i * 4 + j, j * 4 + i, j * 4 + j].map((k) => p.cells[k]);
        S.move(e, p.st.x + 15, -40, at, QUICK, 0);
        S.seq.push([hit, { scale: [1, 1.5, 1] }, { duration: 0.34, at: at + 0.16, ease: "easeOut" }]);
        S.move(e, slot.x, slot.y, at + 0.34, QUICK, 8);
      };
      const [a, o] = parts;
      compute(a, 0.6);
      S.count(0.9, 0, 1);
      a.es.forEach((e, i) => read(a, e, pairs[i % 6], 1.35 + i * 0.3, out(i)));
      S.done(1.35 + (n - 1) * 0.3 + 0.6, 0);
      o.es.forEach((e, i) => {
        const t = 0.6 + i * 0.82;
        compute(o, t);
        S.count(t + 0.3, 1, i + 1);
        read(o, e, pairs[i % 6], t + 0.46, out(i));
        S.seq.push([o.cells, { scale: [1, 0.3], opacity: [1, 0] }, { duration: 0.2, delay: stagger(0.006), at: t + 0.7 }]);
      });
      S.done(0.6 + (n - 1) * 0.82 + 0.95, 1);
      return 0.6 + (n - 1) * 0.82 + 1.2;
    },

    skip(L, S, W, data) {
      const n = data.n;
      const costly = data.costly;
      const queue = pile(0, W * 0.3, n);
      const out = pile(W * 0.64, W, n);
      const parts = L.map((lane) => {
        const gate = lane.who === "andrey" ? station(lane.track, Math.round(W * 0.36), 6, 54, "quick check", true) : null;
        const inv = station(lane.track, Math.round(W * 0.47), 62, 46, "invert");
        const es = edges(lane.track, n);
        es.forEach((e, i) => S.place(e, queue(i).x, queue(i).y));
        return { gate, inv, es };
      });
      const [a, o] = parts;
      const settled = a.es.filter((_, i) => !costly.includes(i));
      a.es.forEach((e, i) => S.move(e, a.gate.x - 34, -16 + (i % 5) * 8, 0.6 + i * 0.03, SETTLE, 0));
      S.glow(a.gate, 1.15);
      S.seq.push([settled.map((e) => e.querySelector(".wf-line")), { opacity: [1, 0] }, { duration: 0.3, at: 1.25 }]);
      settled.forEach((e, k) => {
        S.move(e, out(k).x, out(k).y, 1.4 + k * 0.035, SETTLE, 16);
        S.seq.push([e, { opacity: [1, 0.45] }, { duration: 0.4, at: 1.4 + k * 0.035 }]);
      });
      costly.forEach((i, k) => {
        const t = 1.9 + k * 0.5;
        S.move(a.es[i], a.inv.x + 18, 0, t, QUICK, 10);
        S.glow(a.inv, t + 0.24);
        S.count(t + 0.26, 0, k + 1);
        S.move(a.es[i], out(settled.length + k).x, out(settled.length + k).y, t + 0.32, QUICK, 8);
      });
      S.done(1.9 + (costly.length - 1) * 0.5 + 0.7, 0);
      o.es.forEach((e, i) => {
        const t = 0.6 + i * 0.5;
        S.move(e, o.inv.x + 18, 0, t, QUICK, 10);
        S.glow(o.inv, t + 0.24);
        S.count(t + 0.26, 1, i + 1);
        S.move(e, out(i).x, out(i).y, t + 0.32, QUICK, 8);
      });
      S.done(0.6 + (n - 1) * 0.5 + 0.7, 1);
      return 0.6 + (n - 1) * 0.5 + 1.0;
    },

    hardware(L, S, W, data) {
      const { rows, cols, waves } = data;
      const size = 10;
      const gap = 3;
      const x0 = Math.round((W - cols * (size + gap) + gap) / 2);
      const parts = L.map((lane) => {
        const cells = matrix(lane.track, x0, rows, cols, size, gap);
        cells.forEach((c) => { c.style.opacity = 0; });
        return { cells };
      });
      const [a, o] = parts;
      const width = cols / waves;
      for (let w = 0; w < waves; w++) {
        const at = 0.6 + w * 0.32;
        const wave = a.cells.filter((_, k) => Math.floor((k % cols) / width) === w);
        S.seq.push([wave, { scale: [0.3, 1], opacity: [0, 1] }, { ...QUICK, delay: stagger(0.012), at }]);
        S.count(at + 0.05, 0, w + 1);
      }
      S.done(0.6 + (waves - 1) * 0.32 + 0.45, 0);
      o.cells.forEach((c, k) => {
        const at = 0.6 + k * 0.08;
        S.seq.push([c, { scale: [0.3, 1], opacity: [0, 1] }, { ...QUICK, at }]);
        S.count(at, 1, k + 1);
      });
      S.done(0.6 + (o.cells.length - 1) * 0.08 + 0.35, 1);
      return 0.6 + (o.cells.length - 1) * 0.08 + 0.8;
    },
  };

  function settings(figure) {
    const d = figure.dataset;
    const ints = (v) => (v || "").split(" ").filter(Boolean).map(Number);
    return {
      key: d.scene,
      n: Number(d.n),
      costly: ints(d.costly),
      rows: Number(d.rows),
      cols: Number(d.cols),
      waves: Number(d.waves),
      units: d.unit.split(" "),
    };
  }

  // ---- playing a scene ----------------------------------------------------------------------
  // Returns controls: done (a promise), pause, resume, stop. With reduced motion the scene jumps
  // to its last frame.
  function play(figure, { onProgress, entrance = false, hold = false } = {}) {
    const data = settings(figure);
    const L = lanes(figure);
    const S = score();
    const W = WIDTH;
    fit(figure);
    resizing.observe(figure);
    const end = SCENES[data.key](L, S, W, data);
    if (entrance) {
      const items = L.flatMap((lane) => [...lane.track.querySelectorAll(".wf-edge,.wf-station,.wf-bar,.wf-tag")]);
      S.seq.push([items, { opacity: [0, 1] }, { duration: 0.35, delay: stagger(0.012), at: 0 }]);
    }
    // A linear clock segment so the sequence lasts the whole scene.
    const clock = make("i", "wf-clock", L[0].track);
    clock.style.cssText = "position:absolute;width:0;height:0";
    S.seq.push([clock, { opacity: [0, 1] }, { duration: end, ease: "linear", at: 0 }]);

    const shown = L.map(() => ({ value: 0, done: false, roll: null }));
    const setCount = (k, value) => {
      const lane = L[k];
      const state = shown[k];
      const from = state.value;
      state.value = value;
      // The unit follows the number on screen, so a roll from 0 never reads "0 call".
      const show = (n) => {
        lane.num.textContent = n;
        lane.unit.textContent = n === 1 ? data.units[0] : data.units[1];
      };
      if (state.roll) state.roll.stop();
      if (still) { show(value); return; }
      state.roll = animate(from, value, { duration: 0.28, ease: "easeOut", onUpdate: (v) => show(Math.round(v)) });
    };
    L.forEach((lane, k) => {
      lane.num.textContent = "0";
      lane.unit.textContent = data.units[1];
      lane.badge.style.opacity = 0;
      shown[k].value = 0;
    });

    const controls = animate(S.seq);
    let t = 0;
    let last = performance.now();
    let running = !hold && !still;
    let raf = 0;
    const tick = () => {
      const now = performance.now();
      if (running) t = Math.min(end, t + (now - last) / 1000);
      last = now;
      for (const e of S.events) {
        if (e.at > t) continue;
        const state = shown[e.lane];
        if (e.done && !state.done) {
          state.done = true;
          if (still) L[e.lane].badge.style.opacity = 1;
          else animate(L[e.lane].badge, { opacity: [0, 1], scale: [0.4, 1] }, { type: "spring", bounce: 0.45, visualDuration: 0.35 });
        } else if (!e.done && e.value > state.value) {
          setCount(e.lane, e.value);
        }
      }
      if (onProgress) onProgress(t / end);
    };
    const loop = () => { tick(); raf = requestAnimationFrame(loop); };
    if (still) {
      controls.complete();
      t = end;
      tick();
    } else {
      if (hold) controls.pause();
      loop();
    }
    const done = controls.finished.then(() => { cancelAnimationFrame(raf); t = end; tick(); }, () => {});
    return {
      done,
      pause() { running = false; controls.pause(); },
      resume() { last = performance.now(); running = true; controls.play(); },
      stop() { cancelAnimationFrame(raf); controls.stop(); },
    };
  }

  // ---- in a page: play once in view, then offer a replay -------------------------------------
  for (const figure of document.querySelectorAll('.wf-scene[data-play="view"]')) {
    const replay = figure.querySelector(".wf-replay");
    let run = play(figure, { hold: true });
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      run.resume();
      run.done.then(() => { if (!still) replay.hidden = false; });
    };
    replay.addEventListener("click", () => {
      run.stop();
      replay.hidden = true;
      run = play(figure);
      run.done.then(() => { replay.hidden = false; });
    });
    if (!still) inView(figure, () => { start(); }, { amount: 0.6 });
  }

  // ---- the homepage's dialog: one scene at a time, advancing on its own ----------------------
  for (const dialog of document.querySelectorAll("dialog.wf-dialog")) {
    const panels = [...dialog.querySelectorAll(".wf-panel")];
    const dots = [...dialog.querySelectorAll(".wf-dot")];
    const button = dialog.querySelector(".wf-play");
    let index = 0;
    let run = null;
    let paused = false;
    let finished = false;
    let timer = 0;
    let ticket = 0;
    const setState = (state) => {
      button.dataset.state = state;
      button.setAttribute("aria-label", { playing: "Pause", paused: "Play", ended: "Replay" }[state]);
    };
    const advance = () => {
      if (paused || !dialog.open) return;
      if (index < panels.length - 1) timer = setTimeout(() => show(index + 1), 2200);
      else setState("ended");
    };
    function show(k, { fade = true } = {}) {
      const mine = ++ticket;
      clearTimeout(timer);
      if (run) run.stop();
      const previous = panels[index];
      index = k;
      finished = false;
      dots.forEach((dot, j) => {
        dot.setAttribute("aria-selected", String(j === k));
        dot.tabIndex = j === k ? 0 : -1;
        dot.querySelector("i").style.transform = "scaleX(0)";
      });
      panels.forEach((panel, j) => { panel.hidden = j !== k; });
      if (fade && !still && previous !== panels[k]) {
        animate(panels[k], { opacity: [0, 1], y: [10, 0] }, { type: "spring", bounce: 0, visualDuration: 0.45 });
      }
      const bar = dots[k].querySelector("i");
      run = play(panels[k].querySelector(".wf-scene"), {
        entrance: true,
        onProgress: (f) => { bar.style.transform = `scaleX(${f})`; },
      });
      if (paused) run.pause();
      setState(paused ? "paused" : "playing");
      // With reduced motion nothing advances on its own; the dots move between scenes.
      run.done.then(() => { if (mine !== ticket || still) return; finished = true; advance(); });
    }
    dots.forEach((dot, k) => {
      dot.addEventListener("click", () => { paused = false; show(k); });
      dot.addEventListener("keydown", (event) => {
        const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        const next = (k + step + dots.length) % dots.length;
        dots[next].focus();
        paused = false;
        show(next);
      });
    });
    button.addEventListener("click", () => {
      if (button.dataset.state === "ended") { paused = false; show(0); return; }
      paused = !paused;
      if (paused) { clearTimeout(timer); run.pause(); setState("paused"); return; }
      setState("playing");
      if (finished) advance();
      else run.resume();
    });
    if (still) button.hidden = true;
    dialog.querySelector(".wf-close").addEventListener("click", () => dialog.close());
    // A click on the backdrop, outside the dialog's box, closes it.
    dialog.addEventListener("click", (event) => {
      const box = dialog.getBoundingClientRect();
      const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
      if (!inside) dialog.close();
    });
    dialog.addEventListener("close", () => { clearTimeout(timer); if (run) run.stop(); });
    // The opener is a link to the guide's page; with this script it opens the dialog instead.
    for (const opener of document.querySelectorAll(`[data-wf-open="${dialog.id}"]`)) {
      opener.setAttribute("role", "button");
      opener.setAttribute("aria-haspopup", "dialog");
      opener.addEventListener("click", (event) => {
        event.preventDefault();
        dialog.showModal();
        paused = false;
        if (!still) animate(dialog, { opacity: [0, 1], y: [24, 0], scale: [0.97, 1] }, { type: "spring", bounce: 0.12, visualDuration: 0.45 });
        show(0, { fade: false });
      });
    }
  }
})();
