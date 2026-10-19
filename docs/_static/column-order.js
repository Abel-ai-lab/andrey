// The 0.7.0 post's column-order figure (docs/blog/0.7.0/order.html). "Shuffle the columns" draws a
// new order and shows what PC's column-order rule does with it: the colliders slide into the order
// the rule visits them, each marked oriented or skipped, and the graph changes to the one that
// order returns. "Causal order" goes back. The figure's data-states maps each order to its graph's
// index and its colliders' steps. Without script, the figure shows the causal order.
(() => {
  // Move each child of `list` to its place in `items`, sliding it from where it was.
  const slide = (list, items) => {
    const before = new Map(items.map((item) => [item, item.getBoundingClientRect()]));
    for (const item of items) list.append(item);
    for (const item of items) {
      const was = before.get(item);
      const now = item.getBoundingClientRect();
      item.style.transition = "none";
      item.style.transform = `translate(${was.left - now.left}px, ${was.top - now.top}px)`;
    }
    requestAnimationFrame(() => {
      for (const item of items) {
        item.style.transition = "";
        item.style.transform = "";
      }
    });
  };

  for (const figure of document.querySelectorAll(".co")) {
    const states = JSON.parse(figure.dataset.states);
    const columns = figure.querySelector(".co-columns");
    const queue = figure.querySelector(".co-old .co-queue");
    const graphs = [...figure.querySelectorAll(".co-old .co-graph")];
    const count = figure.querySelector(".co-count");
    const seenCount = figure.querySelector(".co-seen");
    const causal = [...columns.children].map((chip) => chip.dataset.column).join("");
    const seen = new Set();
    let current = causal;

    const show = (order) => {
      const chips = new Map([...columns.children].map((chip) => [chip.dataset.column, chip]));
      slide(columns, [...order].map((c) => chips.get(c)));
      const [index, steps] = states[order].split("|");
      const cards = new Map([...queue.children].map((card) => [card.dataset.collider, card]));
      const items = steps.split(",").map((step) => {
        const card = cards.get(step.slice(0, -1));
        const oriented = step.endsWith("+");
        card.classList.toggle("is-skipped", !oriented);
        card.querySelector(".co-mark").textContent = oriented ? "oriented" : "skipped";
        return card;
      });
      slide(queue, items);
      for (const graph of graphs) graph.hidden = graph.dataset.graph !== index;
      seen.add(index);
      count.textContent = graphs[Number(index)].dataset.count;
      seenCount.textContent = seen.size;
      current = order;
    };

    const shuffle = () => {
      const order = [...current];
      do {
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
      } while (order.join("") === current);
      show(order.join(""));
    };

    seen.add(states[causal].split("|")[0]);
    for (const button of figure.querySelectorAll(".co-actions button")) button.hidden = false;
    figure.querySelector(".co-shuffle").addEventListener("click", shuffle);
    figure.querySelector(".co-reset").addEventListener("click", () => show(causal));
  }
})();
