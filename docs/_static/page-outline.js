// The page's outline follows the reader: the entry for the last heading above the top third of the
// screen is marked, in the right sidebar and in the phone's menu. pydata-sphinx-theme marks an
// entry only when its heading crosses that line, so after a jump, a reload, or a scroll back up
// its mark would be stale; this sets it on every scroll, with the theme's classes. At the page's
// foot the last entry is marked, since a short last section never reaches the line. An entry the
// reader picks stays marked until they scroll again. The right sidebar scrolls itself to keep the
// mark in view.
(() => {
  const lists = [...document.querySelectorAll("#pst-page-toc-nav, .page-outline")];
  const entries = lists
    .flatMap((list) => [...list.querySelectorAll("a.nav-link[href^='#']")])
    .map((link) => {
      const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      return { link, heading: target?.querySelector("h1, h2, h3, h4, h5, h6") ?? target };
    })
    .filter((entry) => entry.heading);
  if (!entries.length) return;
  const sidebar = document.querySelector(".bd-sidebar-secondary");
  let picked = false;
  let frame = 0;

  const show = (heading) => {
    for (const { link, heading: own } of entries) {
      const on = own === heading;
      link.classList.toggle("active", on);
      if (on) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }
    for (const list of lists) {
      for (const item of list.querySelectorAll(".toc-entry")) {
        item.classList.toggle("active", Boolean(item.querySelector("a.nav-link.active")));
      }
    }
    const link = sidebar?.querySelector("a.nav-link.active");
    if (link && sidebar.scrollHeight > sidebar.clientHeight) {
      const box = sidebar.getBoundingClientRect();
      const at = link.getBoundingClientRect();
      if (at.top < box.top || at.bottom > box.bottom) {
        sidebar.scrollTop += at.top - box.top - box.height / 3;
      }
    }
  };

  const follow = () => {
    frame = 0;
    if (picked) return;
    const line = innerHeight * 0.3;
    let current = entries[0].heading;
    for (const { heading } of entries) {
      if (heading.getBoundingClientRect().top <= line) current = heading;
    }
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) {
      current = entries.at(-1).heading;
    }
    show(current);
  };

  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(follow);
  };
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  for (const list of lists) {
    list.addEventListener("click", (event) => {
      const entry = entries.find(({ link }) => link === event.target.closest("a.nav-link"));
      if (!entry) return;
      picked = true;
      show(entry.heading);
    });
  }
  for (const type of ["wheel", "touchstart", "keydown"]) {
    addEventListener(type, () => { picked = false; }, { passive: true });
  }
  schedule();
})();
