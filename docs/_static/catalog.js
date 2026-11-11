// The methods catalog (docs/methods.md). The picker shows one kind of data's panel at a time, and
// choosing a kind sets the Reads filter, which narrows the tables below with the search, the other
// filters, and the experimental switch. Without script, each option links to its panel, which CSS
// shows, and the tables list every method.
(() => {
  const picker = document.querySelector(".kind-picker");
  const filters = document.querySelector(".cat-filters");
  if (!picker || !filters) return;
  const rows = [...document.querySelectorAll(".cat-table tbody tr")];
  const search = filters.querySelector('input[type="search"]');
  const experimental = filters.querySelector('input[type="checkbox"]');
  const menus = [...filters.querySelectorAll("select")];
  const reads = filters.querySelector('select[data-filter="kinds"]');
  const count = filters.querySelector(".cat-count b");
  const empty = filters.querySelector(".cat-empty");
  const options = [...picker.querySelectorAll(".kind-opt")];
  const panels = [...picker.querySelectorAll(".kind-panel")];

  // A method shows when it matches the search and every filter that is set.
  function apply() {
    const text = search.value.trim().toLowerCase();
    let shown = 0;
    for (const row of rows) {
      const fits =
        (!text || row.dataset.search.includes(text)) &&
        (experimental.checked || row.dataset.status === "supported") &&
        menus.every(
          ({ value, dataset: { filter } }) =>
            !value ||
            (filter === "kinds"
              ? row.dataset.kinds.split(" ").includes(value)
              : row.dataset[filter] === value),
        );
      row.hidden = !fits;
      shown += fits;
    }
    for (const table of document.querySelectorAll(".cat-table")) {
      table.closest("section").hidden = !table.querySelector("tbody tr:not([hidden])");
    }
    count.textContent = shown;
    empty.hidden = shown > 0;
  }

  function choose(kind, filter) {
    picker.classList.add("is-live");
    for (const option of options) {
      if (option.dataset.kind === kind) option.setAttribute("aria-current", "true");
      else option.removeAttribute("aria-current");
    }
    for (const panel of panels) panel.hidden = panel.dataset.kind !== kind;
    if (!filter) return;
    reads.value = kind;
    apply();
  }

  for (const menu of menus) menu.addEventListener("change", apply);
  search.addEventListener("input", apply);
  experimental.addEventListener("change", apply);
  for (const option of options) {
    option.addEventListener("click", (event) => {
      event.preventDefault();
      choose(option.dataset.kind, true);
      history.replaceState(null, "", option.getAttribute("href"));
    });
  }
  const targeted = () => panels.find((p) => `#${p.id}` === location.hash);
  addEventListener("hashchange", () => targeted() && choose(targeted().dataset.kind, false));
  choose((targeted() || panels[0]).dataset.kind, false);
  filters.hidden = false;
})();
