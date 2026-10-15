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

  // Scale each lane's canvas to its track, up to the scene's --wf-scale-max, centered when the
  // track is wider.
  function fit(figure) {
    const most = parseFloat(getComputedStyle(figure).getPropertyValue("--wf-scale-max")) || 1.12;
    for (const holder of figure.querySelectorAll(".wf-track")) {
      const canvas = holder.querySelector(".wf-canvas");
      if (!canvas) continue;
      const scale = Math.min(most, holder.clientWidth / WIDTH);
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
  // to its last frame. With scrub, the scene waits at its start and returns its length (end) and
  // seek, which shows it at any time, forward or back.
  function play(figure, { onProgress, entrance = false, hold = false, scrub = false } = {}) {
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
    if (scrub) {
      controls.pause();
      // Each count is the last one its lane has reached by then, and each check shows once its
      // lane is done; the check eases in and out with a transition of its own (why-fast.css).
      const seek = (time) => {
        controls.time = time;
        L.forEach((lane, k) => {
          let value = 0;
          let done = false;
          for (const e of S.events) {
            if (e.lane !== k || e.at > time) continue;
            if (e.done) done = true;
            else value = Math.max(value, e.value);
          }
          lane.num.textContent = value;
          lane.unit.textContent = value === 1 ? data.units[0] : data.units[1];
          lane.badge.style.opacity = done ? 1 : 0;
          lane.badge.style.scale = done ? 1 : 0.4;
        });
      };
      seek(0);
      return { end, seek };
    }
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

  // ---- the homepage's story: one scene at a time, played by the scroll -----------------------
  // Without this script, and with reduced motion, the steps stand one after another, each scene
  // at its last frame. Otherwise, where the section's title and a step fit the screen together,
  // the run (.wf-run) pins its stage (.wf-pin) under the navbar and splits its scroll evenly among
  // the scenes: the step whose share the reader is in shows, and the first PLAYED of that share
  // plays its scene, forward or back; the rest holds the scene's last frame. Where they do not
  // fit, the title scrolls away and the story alone pins; where even that does not fit, the steps
  // stay stacked and each scene plays as it rises through the screen.
  const PLAYED = 0.7;
  for (const story of document.querySelectorAll(".wf-story")) {
    const steps = [...story.querySelectorAll(".wf-step")];
    const figures = steps.map((step) => step.querySelector(".wf-scene"));
    if (still) {
      figures.forEach((figure) => play(figure));
      continue;
    }
    const run = story.closest(".wf-run");
    const stage = run.querySelector(".wf-pin");
    const scenes = figures.map((figure) => play(figure, { scrub: true }));
    const under = () => parseFloat(getComputedStyle(stage).top) || 0; // where the stage pins
    const shown = scenes.map(() => 0);
    run.style.setProperty("--steps", steps.length);

    // A bar per step, which fills as its scene plays and takes the reader to it.
    const rail = document.createElement("div");
    rail.className = "wf-rail";
    const bars = steps.map((step, k) => {
      const button = make("button", "wf-bar-step", rail);
      button.type = "button";
      const title = step.querySelector("h3").textContent;
      button.innerHTML = "<i></i><span></span>";
      button.querySelector("span").textContent = title;
      button.addEventListener("click", () => {
        const box = run.getBoundingClientRect();
        const span = box.height - stage.offsetHeight;
        const top = window.scrollY + box.top - under() + span * ((k + PLAYED) / steps.length);
        window.scrollTo({ top, behavior: "smooth" });
      });
      return button;
    });
    story.prepend(rail);

    // Where the reader is: through the pinned run, from 0 to 1, or, stacked, through each step.
    const pinned = () => run.classList.contains("is-pinned");
    const through = () => {
      const box = run.getBoundingClientRect();
      const span = box.height - stage.offsetHeight;
      return span > 0 ? Math.min(1, Math.max(0, (under() - box.top) / span)) : 0;
    };
    const goals = () => {
      if (!pinned()) {
        return figures.map((figure, k) => {
          const box = figure.getBoundingClientRect();
          const rise = (innerHeight * 0.9 - box.top) / (innerHeight * 0.5);
          return scenes[k].end * Math.min(1, Math.max(0, rise));
        });
      }
      const at = through() * steps.length;
      const current = Math.min(steps.length - 1, Math.floor(at));
      steps.forEach((step, k) => step.classList.toggle("is-on", k === current));
      bars.forEach((bar, k) => {
        if (k === current) bar.setAttribute("aria-current", "step");
        else bar.removeAttribute("aria-current");
        bar.style.setProperty("--fill", Math.min(1, Math.max(0, (at - k) / PLAYED)).toFixed(3));
      });
      return scenes.map((scene, k) =>
        k < current ? scene.end : k > current ? 0 : scene.end * Math.min(1, (at - k) / PLAYED),
      );
    };

    // Each scene eases toward where the scroll puts it, so a step of the wheel still flows.
    let raf = 0;
    const frame = () => {
      raf = 0;
      let moving = false;
      goals().forEach((goal, k) => {
        const next = Math.abs(goal - shown[k]) < 0.01 ? goal : shown[k] + (goal - shown[k]) * 0.22;
        if (next !== shown[k]) scenes[k].seek((shown[k] = next));
        if (next !== goal) moving = true;
      });
      if (moving) raf = requestAnimationFrame(frame);
    };
    const request = () => { if (!raf) raf = requestAnimationFrame(frame); };

    // Pin only where the stage fits under the navbar: with the section's title, or else with the
    // title lifted above the screen's top (--lift), so the story alone stays. The stage's own
    // boxes measure it, not its scroll height, which counts the hidden steps' offset.
    const fit = () => {
      run.classList.add("is-pinned");
      run.style.setProperty("--lift", "0px");
      const fits = () =>
        story.getBoundingClientRect().bottom - stage.getBoundingClientRect().top <=
        parseFloat(getComputedStyle(stage).maxHeight) + 1;
      if (!fits()) {
        const lift = story.getBoundingClientRect().top - stage.getBoundingClientRect().top;
        run.style.setProperty("--lift", `${Math.round(lift)}px`);
      }
      const pinned = fits();
      run.style.setProperty("--pin-h", `${stage.offsetHeight}px`);
      run.classList.toggle("is-pinned", pinned);
      if (!pinned) steps.forEach((step) => step.classList.remove("is-on"));
      request();
    };
    fit();
    // The web font arrives after the first measure and rewraps the text, so measure again.
    if (document.fonts) document.fonts.ready.then(fit);
    addEventListener("scroll", request, { passive: true });
    let resized = 0;
    addEventListener("resize", () => { clearTimeout(resized); resized = setTimeout(fit, 150); });
  }
})();
