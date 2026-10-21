// The homepage's bento tiles. Each tile, or a note's link, is a link to its details further down
// the page; with this script it opens the same details as a dialog instead. Enhancement only:
// without it every dialog's content is shown in the page, and a link jumps to it.
(() => {
  const dialogs = [...document.querySelectorAll("dialog.tile-dialog")];
  const open = (dialog) => { if (!dialog.open) dialog.showModal(); };

  for (const link of document.querySelectorAll("a[data-tile]")) {
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
  const page = document.querySelector(".andrey-home main");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (page && !still && !CSS.supports("animation-timeline: view()") && "IntersectionObserver" in window) {
    page.classList.add("is-flying");
    const arrive = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-in");
        arrive.unobserve(entry.target);
      }
    }, { rootMargin: "0px 0px -10% 0px" });
    for (const tile of page.querySelectorAll(".bento-grid > .tile")) arrive.observe(tile);
  }

  // The art bands' dim where the browser has no view timeline: dimmed until most of the band is
  // on the screen.
  if (!still && !CSS.supports("animation-timeline: view()") && "IntersectionObserver" in window) {
    const lit = new IntersectionObserver((entries) => {
      for (const entry of entries) entry.target.classList.toggle("is-dimmed", entry.intersectionRatio < 0.6);
    }, { threshold: [0, 0.6] });
    for (const art of document.querySelectorAll(".cover-art")) lit.observe(art);
  }

  // A cover's sweep runs while the cover is on the screen; under reduced motion it never shows.
  if (!still && "IntersectionObserver" in window) {
    const lit = new IntersectionObserver((entries) => {
      for (const entry of entries) entry.target.toggleAttribute("data-on", entry.isIntersecting);
    });
    for (const sweep of document.querySelectorAll(".cover-sweep")) lit.observe(sweep);
  }

  // A segmented panel: its tabs show one panel at a time; the arrow keys move between tabs.
  const choose = new Map();
  for (const seg of document.querySelectorAll("[data-seg]")) {
    const tabs = [...seg.querySelectorAll('[role="tab"]')];
    let show = (tab, focus) => {
      for (const each of tabs) {
        const on = each === tab;
        each.setAttribute("aria-selected", String(on));
        each.tabIndex = on ? 0 : -1;
        document.getElementById(each.getAttribute("aria-controls")).hidden = !on;
      }
      if (focus) tab.focus();
    };
    // A pill glides under the tab beneath the pointer and settles back on the chosen tab.
    const bar = seg.querySelector('[role="tablist"]');
    const glide = document.createElement("span");
    glide.className = "seg-glide";
    glide.setAttribute("aria-hidden", "true");
    bar.prepend(glide);
    bar.classList.add("has-glide");
    const chosen = () => tabs.find((tab) => tab.getAttribute("aria-selected") === "true");
    const light = (tab) => {
      glide.style.width = `${tab.offsetWidth}px`;
      glide.style.transform = `translateX(${tab.offsetLeft}px)`;
      for (const each of tabs) each.classList.toggle("is-lit", each === tab);
    };
    for (const tab of tabs) tab.addEventListener("pointerenter", () => light(tab));
    bar.addEventListener("pointerleave", () => light(chosen()));
    window.addEventListener("resize", () => light(chosen()));
    const settle = show;
    show = (tab, focus) => { settle(tab, focus); light(tab); };
    show(chosen() || tabs[0], false);
    tabs.forEach((tab, i) => {
      choose.set(tab.getAttribute("aria-controls"), () => show(tab, false));
      tab.addEventListener("click", () => show(tab, false));
      tab.addEventListener("keydown", (event) => {
        const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        show(tabs[(i + step + tabs.length) % tabs.length], true);
      });
    });
  }

  // An address that ends in a tile's id, such as an old link to #methods, opens that tile; one
  // that ends in a panel's id, such as #speed-score, shows that panel.
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (target && target.matches("dialog.tile-dialog")) open(target);
  if (target && choose.has(target.id)) {
    choose.get(target.id)();
    target.scrollIntoView();
  }
})();
