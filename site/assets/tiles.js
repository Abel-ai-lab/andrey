// The homepage's bento tiles. Each tile is a link to its details further down the page; with this
// script it opens the same details as a dialog instead. Enhancement only: without it every
// dialog's content is shown in the page, and a tile jumps to it.
(() => {
  const dialogs = [...document.querySelectorAll("dialog.tile-dialog")];
  const open = (dialog) => { if (!dialog.open) dialog.showModal(); };

  for (const link of document.querySelectorAll("a.tile-face[data-tile]")) {
    const dialog = document.getElementById(link.dataset.tile);
    if (!dialog) continue;
    link.setAttribute("aria-haspopup", "dialog");
    link.addEventListener("click", (event) => {
      event.preventDefault();
      open(dialog);
    });
  }

  for (const dialog of dialogs) {
    dialog.querySelector(".wf-close").addEventListener("click", () => dialog.close());
    // A click on the backdrop, outside the dialog's box, closes it.
    dialog.addEventListener("click", (event) => {
      const box = dialog.getBoundingClientRect();
      const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
      if (!inside) dialog.close();
    });
    // A link to a note further down the page closes the dialog and goes there.
    for (const link of dialog.querySelectorAll('a[href^="#"]')) {
      link.addEventListener("click", () => dialog.close());
    }
  }

  // The fly-in where the browser has no view timeline: tiles start hidden only once this runs.
  const bento = document.querySelector(".bento");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (bento && !still && !CSS.supports("animation-timeline: view()") && "IntersectionObserver" in window) {
    bento.classList.add("is-flying");
    const arrive = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-in");
        arrive.unobserve(entry.target);
      }
    }, { rootMargin: "0px 0px -10% 0px" });
    for (const tile of bento.querySelectorAll(".bento-grid > .tile")) arrive.observe(tile);
  }

  // The art bands' dim where the browser has no view timeline: dimmed until most of the band is
  // on the screen.
  if (!still && !CSS.supports("animation-timeline: view()") && "IntersectionObserver" in window) {
    const lit = new IntersectionObserver((entries) => {
      for (const entry of entries) entry.target.classList.toggle("is-dimmed", entry.intersectionRatio < 0.6);
    }, { threshold: [0, 0.6] });
    for (const art of document.querySelectorAll(".cover-art")) lit.observe(art);
  }

  // The figures' carousel: controls, and a slow turn that holds under the pointer or focus, out
  // of view, and under reduced motion. Without the script the row scrolls sideways.
  for (const root of document.querySelectorAll(".carousel")) {
    const track = root.querySelector(".carousel-track");
    const slides = [...track.children];
    const controls = root.querySelector(".carousel-controls");
    const dotsBox = root.querySelector(".carousel-dots");
    let index = 0;
    let held = still;
    const dots = slides.map((_, i) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "carousel-dot";
      dot.setAttribute("aria-label", `Figure ${i + 1}`);
      dot.addEventListener("click", () => go(i));
      dotsBox.append(dot);
      return dot;
    });
    const mark = () => dots.forEach((dot, i) => dot.setAttribute("aria-current", String(i === index)));
    const go = (i) => {
      index = (i + slides.length) % slides.length;
      track.scrollTo({ left: slides[index].offsetLeft - slides[0].offsetLeft, behavior: still ? "auto" : "smooth" });
      mark();
    };
    controls.hidden = false;
    mark();
    for (const button of root.querySelectorAll(".carousel-step")) {
      button.addEventListener("click", () => go(index + Number(button.dataset.step)));
    }
    let settle;
    track.addEventListener("scroll", () => {
      clearTimeout(settle);
      settle = setTimeout(() => {
        const left = track.scrollLeft + slides[0].offsetLeft;
        index = slides.reduce((best, s, i) => (Math.abs(s.offsetLeft - left) < Math.abs(slides[best].offsetLeft - left) ? i : best), 0);
        mark();
      }, 120);
    });
    let visible = false;
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(root);
    const hold = () => { held = true; };
    const release = () => { held = still; };
    root.addEventListener("pointerenter", hold);
    root.addEventListener("pointerleave", release);
    root.addEventListener("focusin", hold);
    root.addEventListener("focusout", (event) => { if (!root.contains(event.relatedTarget)) release(); });
    setInterval(() => { if (!held && visible && !document.hidden) go(index + 1); }, 6000);
  }

  // An address that ends in a tile's id, such as an old link to #methods, opens that tile.
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (target && target.matches("dialog.tile-dialog")) open(target);
})();
