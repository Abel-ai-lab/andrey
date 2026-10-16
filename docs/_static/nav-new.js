// The dot on the navbar's Release cards announces a new release: it shows for a week after the
// release's post (`data-date`), and only until the reader opens the release cards, which this
// browser's local storage records; nothing is sent anywhere. The navbar and the phone's drawer each
// draw the link, so every dot on the page follows the same record. Without storage the dot stays
// hidden.
(() => {
  const dots = document.querySelectorAll(".nav-new[data-release]");
  if (!dots.length) return;
  const release = dots[0].dataset.release;
  const WEEK = 7 * 24 * 60 * 60 * 1000;
  const fresh = Date.now() < Date.parse(`${dots[0].dataset.date}T00:00:00Z`) + WEEK;
  const key = "andrey-release-seen";
  const onCards = Boolean(dots[0].closest(".nav-item.current"));
  try {
    if (onCards) localStorage.setItem(key, release);
    else if (fresh && localStorage.getItem(key) !== release) for (const dot of dots) dot.hidden = false;
  } catch {
    // Storage is blocked: a dot that could never clear is worse than none.
  }
})();
