// The methods catalog (docs/methods.md). The picker shows one kind of data's panel at a time, and
// choosing a kind presses its chip in the filters, which narrow the tables below. Without script,
// each option links to its panel, which CSS shows, and the tables list every method.
(() => {
  const picker = document.querySelector(".kind-picker");
  const filters = document.querySelector(".cat-filters");
  if (!picker || !filters) return;
  const rows = [...document.querySelectorAll(".cat-table tbody tr")];
  const search = filters.querySelector('input[type="search"]');
  const experimental = filters.querySelector('input[type="checkbox"]');
  const count = filters.querySelector(".cat-count b");
  const empty = filters.querySelector(".cat-empty");
  const chips = [...filters.querySelectorAll(".cat-chip")];
  const options = [...picker.querySelectorAll(".kind-opt")];
  const panels = [...picker.querySelectorAll(".kind-panel")];

  // A method shows when it matches the search and, in each chip group, any pressed chip.
  function apply() {
    const pressed = {};
    for (const chip of chips) {
      if (chip.getAttribute("aria-pressed") === "true") {
        (pressed[chip.dataset.filter] ||= []).push(chip.dataset.value);
      }
    }
    const text = search.value.trim().toLowerCase();
    let shown = 0;
    for (const row of rows) {
      const fits =
        (!text || row.dataset.search.includes(text)) &&
        (experimental.checked || row.dataset.status === "supported") &&
        Object.entries(pressed).every(([key, values]) =>
          values.some((v) =>
            key === "kinds" ? row.dataset.kinds.split(" ").includes(v) : row.dataset[key] === v,
          ),
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

  function choose(kind, press) {
    picker.classList.add("is-live");
    for (const option of options) {
      if (option.dataset.kind === kind) option.setAttribute("aria-current", "true");
      else option.removeAttribute("aria-current");
    }
    for (const panel of panels) panel.hidden = panel.dataset.kind !== kind;
    if (!press) return;
    for (const chip of chips) {
      if (chip.dataset.filter !== "kinds") continue;
      chip.setAttribute("aria-pressed", chip.dataset.value === kind);
    }
    apply();
  }

  for (const chip of chips) {
    chip.addEventListener("click", () => {
      chip.setAttribute("aria-pressed", chip.getAttribute("aria-pressed") !== "true");
      apply();
    });
  }
  search.addEventListener("input", apply);
  experimental.addEventListener("change", apply);

  const narrow = matchMedia("(max-width: 1000px)");
  const still = matchMedia("(prefers-reduced-motion: reduce)");
  for (const option of options) {
    option.addEventListener("click", (event) => {
      event.preventDefault();
      choose(option.dataset.kind, true);
      history.replaceState(null, "", option.getAttribute("href"));
      // On a narrow screen the panel sits under the options.
      const panel = panels.find((p) => !p.hidden);
      if (narrow.matches) panel.scrollIntoView({ behavior: still.matches ? "auto" : "smooth" });
    });
  }
  const targeted = () => panels.find((p) => `#${p.id}` === location.hash);
  addEventListener("hashchange", () => targeted() && choose(targeted().dataset.kind, false));
  choose((targeted() || panels[0]).dataset.kind, false);
  filters.hidden = false;
})();
