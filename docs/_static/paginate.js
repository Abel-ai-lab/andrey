// The blog's earlier posts and the Release cards page's earlier releases show PAGE rows at a time,
// with a pager under them. The page is in the address (`?page=2`), so a link or a reload keeps it.
// Without scripts every row shows. The blog's order button starts the list over at its first page.
(() => {
  const PAGE = 9;
  const blog = document.getElementById("blog-earlier");
  const releases = document.querySelector("section#earlier-releases");
  const rows = blog
    ? () => [...blog.children]
    : releases && (() => [...releases.querySelectorAll(":scope > .release-row")]);
  if (!rows || rows().length <= PAGE) return;
  const pages = Math.ceil(rows().length / PAGE);
  const head = (blog || releases).closest("section") || blog || releases;
  const pager = document.createElement("nav");
  pager.className = "pager";
  pager.setAttribute("aria-label", blog ? "Earlier posts, pages" : "Earlier releases, pages");
  rows().at(-1).after(pager);

  const link = (page, label, rel) => {
    const a = document.createElement("a");
    const url = new URL(location.href);
    if (page === 1) url.searchParams.delete("page");
    else url.searchParams.set("page", page);
    a.href = url.pathname + url.search;
    a.textContent = label;
    a.dataset.page = page;
    if (rel) a.rel = rel;
    return a;
  };

  const show = (page, moved) => {
    page = Math.min(Math.max(page, 1), pages);
    rows().forEach((row, i) => {
      row.hidden = Math.floor(i / PAGE) + 1 !== page;
    });
    // The pager sits after the list's last row, which a reversed order moves.
    rows().at(-1).after(pager);
    pager.replaceChildren();
    const previous = link(page - 1, "Previous", "prev");
    const next = link(page + 1, "Next", "next");
    previous.classList.add("pager-step");
    next.classList.add("pager-step");
    if (page === 1) previous.setAttribute("aria-disabled", "true");
    if (page === pages) next.setAttribute("aria-disabled", "true");
    const numbers = document.createElement("ol");
    for (let n = 1; n <= pages; n++) {
      const item = document.createElement("li");
      const a = link(n, String(n));
      a.setAttribute("aria-label", `Page ${n}`);
      if (n === page) a.setAttribute("aria-current", "page");
      item.append(a);
      numbers.append(item);
    }
    pager.append(previous, numbers, next);
    if (moved) {
      history.pushState({ page }, "", pager.querySelector(`[data-page="${page}"]`).href);
      head.scrollIntoView({ block: "start" });
    }
  };

  pager.addEventListener("click", (event) => {
    const a = event.target.closest("a[data-page]");
    if (!a) return;
    event.preventDefault();
    if (a.getAttribute("aria-disabled") === "true" || a.hasAttribute("aria-current")) return;
    show(Number(a.dataset.page), true);
  });
  window.addEventListener("popstate", () => {
    show(Number(new URL(location.href).searchParams.get("page")) || 1, false);
  });
  document.querySelector(".blog-order")?.addEventListener("click", () => {
    const url = new URL(location.href);
    url.searchParams.delete("page");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
    show(1, false);
  });
  show(Number(new URL(location.href).searchParams.get("page")) || 1, false);
})();
