/* Sphinx loads this script in the head; enhance once the guide markup is ready. */
document.addEventListener("DOMContentLoaded", () => {
  /* Progressive enhancement: every example remains readable when JavaScript is disabled. */
  document.querySelectorAll("[data-graph-explorer]").forEach((explorer) => {
    const controls = explorer.querySelector(".graph-controls");
    const examples = explorer.querySelectorAll("[data-example]");
    const select = () => {
      const selected = controls.querySelector("input:checked").value;
      examples.forEach((example) => { example.hidden = example.dataset.example !== selected; });
    };
    controls.hidden = false;
    explorer.querySelector(".graph-help").hidden = false;
    controls.addEventListener("change", select);
    select();

    examples.forEach((example) => {
      const cells = example.querySelectorAll("td[data-row]");
      const clear = () => {
        cells.forEach((cell) => cell.classList.remove("graph-linked"));
        example.querySelector(".graph-pair").classList.remove("graph-linked");
      };
      const highlight = (active) => {
        clear();
        cells.forEach((cell) => {
          const same = cell.dataset.row === active.dataset.row &&
            cell.dataset.col === active.dataset.col;
          const reverse = cell.dataset.row === active.dataset.col &&
            cell.dataset.col === active.dataset.row;
          cell.classList.toggle("graph-linked", same || reverse);
        });
        example.querySelector(".graph-pair").classList.toggle(
          "graph-linked", active.dataset.row !== active.dataset.col,
        );
      };
      cells.forEach((cell) => {
        cell.tabIndex = 0;
        cell.addEventListener("focus", () => highlight(cell));
        cell.addEventListener("mouseenter", () => highlight(cell));
        cell.addEventListener("blur", clear);
        cell.addEventListener("mouseleave", () => {
          clear();
          if (example.contains(document.activeElement) &&
              document.activeElement.matches("td[data-row]")) {
            highlight(document.activeElement);
          }
        });
      });
    });
  });
});
