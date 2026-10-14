// The homepage's bento tiles. Each tile is a link to its details further down the page; with this
// script it opens the same details as a dialog instead. Enhancement only: without it every
// dialog's content is shown in the page, and a tile jumps to it.
(() => {
  const dialogs = [...document.querySelectorAll("dialog.tile-dialog")];
  const open = (dialog) => { if (!dialog.open) dialog.showModal(); };

  for (const link of document.querySelectorAll("a.tile-face[data-tile]")) {
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

  // An address that ends in a tile's id, such as an old link to #methods, opens that tile.
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (target && target.matches("dialog.tile-dialog")) open(target);
})();
