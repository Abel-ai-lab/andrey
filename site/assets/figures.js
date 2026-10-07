// The timing comparison and the speed charts, on the homepage and in the launch post, and the
// homepage's speedup panel and agent session. Enhancements only; every figure is complete without
// this script.
(() => {
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  // The speed and accuracy belts: the script drives them so they can be swiped. A belt advances
  // at a steady pace and stops under the pointer or focus. It follows a drag with the mouse or a
  // finger, glides on after a flick, and moves with a sideways trackpad swipe or Shift and the
  // wheel; a vertical swipe still scrolls the page. The loop wraps where the hidden copy begins.
  // Under reduced motion a belt is a row to scroll by hand.
  for (const belt of document.querySelectorAll(".tc-belt")) {
    const track = belt.querySelector(".tc-track");
    const copy = track && track.querySelector(".tc-copy");
    if (!copy || motion.matches) continue;
    belt.classList.add("is-driven");
    const pace = 50; // pixels a second
    let offset = 0;
    let last = null;
    let paused = false;
    let drag = null;
    let glide = 0; // pixels a millisecond, after a flick
    const wrap = () => {
      const loop = copy.offsetLeft;
      if (loop > 0) offset = ((offset % loop) + loop) % loop;
    };
    const draw = () => { track.style.transform = "translateX(" + -offset + "px)"; };
    const frame = (time) => {
      const step = last === null ? 0 : Math.min(time - last, 64);
      if (drag) {
        // the finger or mouse sets the offset
      } else if (glide) {
        offset += glide * step;
        glide *= Math.pow(0.94, step / 16);
        if (Math.abs(glide) < 0.02) glide = 0;
      } else if (!paused) {
        offset += (pace * step) / 1000;
      }
      last = time;
      wrap();
      draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    belt.addEventListener("pointerenter", (event) => { if (event.pointerType === "mouse") paused = true; });
    belt.addEventListener("pointerleave", () => { paused = false; });
    belt.addEventListener("focusin", () => { paused = true; });
    belt.addEventListener("focusout", () => { paused = false; });
    belt.addEventListener("pointerdown", (event) => {
      glide = 0;
      drag = { x: event.clientX, from: offset, trail: [[event.timeStamp, offset]] };
      belt.setPointerCapture(event.pointerId);
      belt.classList.add("is-dragging");
    });
    belt.addEventListener("pointermove", (event) => {
      if (!drag) return;
      offset = drag.from - (event.clientX - drag.x);
      drag.trail = [...drag.trail.filter(([t]) => event.timeStamp - t < 100), [event.timeStamp, offset]];
      wrap();
      draw();
    });
    const release = (event) => {
      if (!drag) return;
      // The flick's speed over its last tenth of a second carries on, and slows to a stop.
      const [t0, x0] = drag.trail[0];
      const elapsed = event.timeStamp - t0;
      glide = elapsed > 0 && event.type === "pointerup" ? (offset - x0) / elapsed : 0;
      drag = null;
      belt.classList.remove("is-dragging");
    };
    belt.addEventListener("pointerup", release);
    belt.addEventListener("pointercancel", release);
    belt.addEventListener("wheel", (event) => {
      const sideways = event.shiftKey ? event.deltaX || event.deltaY : event.deltaX;
      if (!sideways || (!event.shiftKey && Math.abs(event.deltaX) <= Math.abs(event.deltaY))) return;
      event.preventDefault();
      glide = 0;
      offset += sideways * (event.deltaMode === 1 ? 16 : 1); // lines, in Firefox
      wrap();
      draw();
    }, { passive: false });
  }
  // Hidden panels keep their place, so a newly shown panel restarts its bars instead of showing
  // them done.
  const restart = (race) => {
    if (motion.matches) return;
    for (const animation of race.getAnimations({ subtree: true })) {
      animation.cancel();
      animation.play();
    }
  };
  // The speedup panel plays through its methods every few seconds, from the moment it first comes
  // into view (at once in the hero, on scrolling in the launch post). It holds under the pointer or
  // focus, and stops for good once a reader picks one. Reduced motion keeps the first method.
  for (const race of document.querySelectorAll(".race--speedup")) {
    const choices = [...race.querySelectorAll(".race-choice")];
    if (motion.matches || choices.length < 2) continue;
    let paused = false;
    let stopped = false;
    let timer;
    const step = () => {
      if (paused || document.hidden) return;
      const at = choices.findIndex((choice) => choice.checked);
      choices[(at + 1) % choices.length].checked = true;
      restart(race);
    };
    const seen = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      seen.disconnect();
      if (stopped) return;
      restart(race);
      timer = setInterval(step, 4500);
    }, { threshold: 0.5 });
    seen.observe(race);
    // A reader's pick stops it: a change from the keyboard, or a click, which also covers
    // clicking the method already shown. Setting .checked from here fires neither.
    const stop = () => {
      stopped = true;
      clearInterval(timer);
    };
    race.addEventListener("change", stop);
    race.addEventListener("click", (event) => {
      if (event.target.closest(".race-choice, .race-choice + label")) stop();
    });
    race.addEventListener("pointerenter", () => { paused = true; });
    race.addEventListener("pointerleave", () => { paused = false; });
    race.addEventListener("focusin", () => { paused = true; });
    race.addEventListener("focusout", () => { paused = false; });
  }
  for (const race of document.querySelectorAll(".race")) {
    race.addEventListener("change", () => restart(race));
  }
  // The agent session types each command, then shows its answer and lights its step; the graph
  // comes last. It plays once when it scrolls into view, and again from Replay.
  const wait = (ms) => new Promise((done) => setTimeout(done, ms));
  for (const demo of document.querySelectorAll(".agent-demo")) {
    if (motion.matches) continue;
    const frames = [...demo.querySelectorAll(".frame")];
    const steps = [...demo.querySelectorAll(".agent-step")];
    const graph = demo.querySelector(".agent-graph");
    const replay = demo.querySelector(".term-replay");
    let run = 0;
    const reset = () => {
      demo.classList.add("is-playing");
      for (const item of [...frames, graph]) item.classList.add("is-later");
      for (const step of steps) step.classList.remove("is-on");
    };
    // The untyped rest keeps its place, so a command that wraps holds its lines while it types.
    const type = async (command, mine) => {
      const text = (command.dataset.text ??= command.textContent);
      const typed = document.createElement("span");
      const rest = document.createElement("span");
      rest.className = "untyped";
      command.replaceChildren(typed, rest);
      for (let at = 0; at <= text.length; at++) {
        typed.textContent = text.slice(0, at);
        rest.textContent = text.slice(at);
        await wait(24);
        if (mine !== run) return false;
      }
      return true;
    };
    const play = async () => {
      const mine = ++run;
      reset();
      for (const [i, frame] of frames.entries()) {
        frame.classList.remove("is-later");
        frame.classList.add("is-typing");
        steps.forEach((step, k) => step.classList.toggle("is-on", k === i));
        if (!(await type(frame.querySelector(".cmd-text"), mine))) return;
        await wait(350);
        if (mine !== run) return;
        frame.classList.remove("is-typing");
        await wait(frame.querySelectorAll(".ln").length * 45 + 1100);
        if (mine !== run) return;
      }
      graph.classList.remove("is-later");
      for (const step of steps) step.classList.add("is-on");
    };
    replay.hidden = false;
    replay.addEventListener("click", play);
    reset();
    new IntersectionObserver((entries, observer) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        play();
      }
    }, { threshold: 0.35 }).observe(demo);
  }
})();
