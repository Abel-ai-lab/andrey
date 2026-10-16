// The dot on the navbar's Release cards: shown until the reader opens the release cards, and shown
// again when a newer release is out. The navbar and the phone's drawer each draw the link, so
// every dot on the page follows the same record. Without storage the dot stays hidden.
(() => {
  const dots = document.querySelectorAll(".nav-new[data-release]");
  if (!dots.length) return;
  const release = dots[0].dataset.release;
  const key = "andrey-release-seen";
  const onCards = Boolean(dots[0].closest(".nav-item.current"));
  try {
    if (onCards) localStorage.setItem(key, release);
    else if (localStorage.getItem(key) !== release) for (const dot of dots) dot.hidden = false;
  } catch {
    // Storage is blocked: a dot that could never clear is worse than none.
  }
})();
