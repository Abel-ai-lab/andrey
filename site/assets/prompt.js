// The hero's pill copies the prompt that installs Andrey's agent skill, to paste into a coding
// agent. Enhancement only: without this script, or where the clipboard refuses, the pill leads to
// the FAQ answer that gives the prompt.
(() => {
  const pill = document.querySelector("a.agent-pill[data-prompt]");
  if (!pill || !navigator.clipboard) return;
  const label = pill.querySelector(".agent-pill__label");
  const rest = label.innerHTML;
  let reset;
  pill.setAttribute("role", "button");
  label.setAttribute("aria-live", "polite");
  const copy = (event) => {
    event.preventDefault();
    navigator.clipboard.writeText(pill.dataset.prompt).then(
      () => {
        pill.classList.add("is-copied");
        // The second half, like the label's, is dropped on a phone (site.css).
        label.innerHTML =
          'Prompt copied<span class="agent-pill__more">. Paste it into your agent</span>';
        clearTimeout(reset);
        reset = setTimeout(() => {
          pill.classList.remove("is-copied");
          label.innerHTML = rest;
        }, 2500);
      },
      () => { window.location.href = pill.href; },
    );
  };
  pill.addEventListener("click", copy);
  // A button also answers the space bar.
  pill.addEventListener("keydown", (event) => { if (event.key === " ") copy(event); });
})();
