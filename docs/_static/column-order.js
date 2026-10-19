// The 0.7.0 post's column-order figure (docs/blog/0.7.0/order.html). The column-order rule's board
// holds a column per data column, each collider under the column it points into, so the rule's
// visiting order reads left to right. "Shuffle the columns" draws a new order, a click on a column
// moves it first, and "Original order" goes back: the columns slide into place, each collider is
// marked as orienting or as skipped for an edge an earlier one took, and the graph changes to the
// one that order returns. The figure's data-states maps each order to its graph's index and its
// colliders' steps. Without script, the figure shows the original order.
(() => {
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // Move each item to its place in `list`, in the order of `items`, sliding it from where it was.
  const slide = (list, items) => {
    const before = new Map(items.map((item) => [item, item.getBoundingClientRect()]));
    for (const item of items) list.append(item);
    if (motion.matches) return;
    for (const item of items) {
      const was = before.get(item);
      const now = item.getBoundingClientRect();
      item.style.transition = "none";
      item.style.transform = `translate(${was.left - now.left}px, ${was.top - now.top}px)`;
    }
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        for (const item of items) {
          item.style.transition = "";
          item.style.transform = "";
        }
      }),
    );
  };

  // A collider x -> y <- z claims the arrowheads at y on x - y and z - y; a later collider that
  // wants the opposite arrowhead on one of those edges is skipped.
  const replay = (steps) => {
    const heads = new Set();
    return steps.map(([key, oriented], i) => {
      const [x, y, z] = key.split(" ");
      const taken = [x, z].filter((a) => heads.has(`${y}>${a}`)).map((a) => [a, y].sort().join("-"));
      if (oriented) for (const a of [x, z]) heads.add(`${a}>${y}`);
      return { key, oriented, taken, step: i + 1 };
    });
  };

  for (const figure of document.querySelectorAll(".co")) {
    const states = JSON.parse(figure.dataset.states);
    const board = figure.querySelector(".co-board");
    const columns = new Map([...board.children].map((col) => [col.dataset.column, col]));
    const cards = new Map(
      [...board.querySelectorAll(".co-card")].map((card) => [card.dataset.collider, card]),
    );
    const graphs = [...figure.querySelectorAll(".co-old .co-graph")];
    const why = figure.querySelector(".co-why");
    const count = figure.querySelector(".co-count");
    const seenCount = figure.querySelector(".co-seen");
    const original = [...columns.keys()].join("");
    const seen = new Set([states[original].split("|")[0]]);
    let current = original;

    const show = (order) => {
      const [index, steps] = states[order].split("|");
      const verdicts = replay(steps.split(",").map((s) => [s.slice(0, -1), s.endsWith("+")]));
      slide(board, [...order].map((c) => columns.get(c)));
      for (const v of verdicts) {
        const card = cards.get(v.key);
        card.classList.toggle("is-skipped", !v.oriented);
        card.querySelector(".co-step").textContent = v.step;
        card.querySelector(".co-verdict").innerHTML = v.oriented
          ? "orients"
          : `skipped: <code>${v.taken.join(", ")}</code> taken`;
      }
      // Within a column, the cards take the order the rule visits them.
      for (const col of columns.values()) {
        const list = col.querySelector(".co-cards");
        const mine = verdicts.filter((v) => v.key.split(" ")[1] === col.dataset.column);
        if (mine.length > 1) slide(list, mine.map((v) => cards.get(v.key)));
      }
      for (const graph of graphs) graph.hidden = graph.dataset.graph !== index;
      const graph = graphs[Number(index)];
      why.textContent = graph.dataset.why;
      count.textContent = graph.dataset.count;
      seen.add(index);
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

    for (const hidden of figure.querySelectorAll(".co-actions button, .co-hint")) {
      hidden.hidden = false;
    }
    for (const [column, col] of columns) {
      col.querySelector(".co-chip").addEventListener("click", () => {
        show(column + current.replace(column, ""));
      });
    }
    figure.querySelector(".co-shuffle").addEventListener("click", shuffle);
    figure.querySelector(".co-reset").addEventListener("click", () => show(original));
  }
})();
