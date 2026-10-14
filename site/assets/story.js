// The hero's scroll story: "Causal discovery," stays while the last word changes from accelerated
// to accurate to agentic, and the graph changes with it. Enhancement only: without this script,
// or under reduced motion, the hero is one static line and the settled graph.
//
// Where the browser has scroll-driven animations, the stylesheet moves everything from the scroll
// position and this script only turns the story on. Elsewhere it sets data-step on the hero,
// from which of three equal stretches of the track is at the middle of the screen.
(() => {
  const hero = document.querySelector(".hero");
  const track = hero && hero.querySelector(".story-track");
  if (!track || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  hero.classList.add("is-story");
  if (CSS.supports("animation-timeline: scroll()")) return;

  hero.dataset.step = "1";
  const steps = new Map();
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) steps.set(entry.target, entry.isIntersecting);
      const live = [...steps].filter(([, on]) => on).map(([mark]) => Number(mark.dataset.step));
      if (live.length) hero.dataset.step = String(Math.max(...live));
    },
    { rootMargin: "-50% 0px -50% 0px" },
  );
  for (const step of [1, 2, 3]) {
    const mark = document.createElement("div");
    mark.dataset.step = String(step);
    mark.setAttribute("aria-hidden", "true");
    mark.style.cssText = `position:absolute;left:0;width:1px;top:${(step - 1) * 33.3}%;height:33.4%;pointer-events:none`;
    track.append(mark);
    observer.observe(mark);
  }
  track.style.position = "relative";

  // The dim as the cover leaves: on once the bento's top passes the screen's lower fifth.
  const next = hero.nextElementSibling;
  if (next) {
    new IntersectionObserver(
      ([entry]) => {
        const passed = entry.isIntersecting || entry.boundingClientRect.top < 0;
        hero.toggleAttribute("data-leaving", passed);
      },
      { rootMargin: "0px 0px -20% 0px" },
    ).observe(next);
  }
})();
