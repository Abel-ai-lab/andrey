// The hero's pill copies the prompt that installs Andrey's agent skill, to paste into a coding
// agent. Enhancement only: without this script, or where the clipboard refuses, the pill leads to
// the FAQ answer that gives the prompt.
(() => {
  const pill = document.querySelector("a.agent-pill[data-prompt]");
  if (!pill || !navigator.clipboard) return;
  const label = pill.querySelector("span:first-child");
  const text = label.textContent;
  let reset;
  pill.setAttribute("role", "button");
  label.setAttribute("aria-live", "polite");
  const copy = (event) => {
    event.preventDefault();
    navigator.clipboard.writeText(pill.dataset.prompt).then(
      () => {
        pill.classList.add("is-copied");
        label.textContent = "Copied. Paste it into your agent";
        clearTimeout(reset);
        reset = setTimeout(() => {
          pill.classList.remove("is-copied");
          label.textContent = text;
        }, 2500);
      },
      () => { window.location.href = pill.href; },
    );
  };
  pill.addEventListener("click", copy);
  // A button also answers the space bar.
  pill.addEventListener("keydown", (event) => { if (event.key === " ") copy(event); });
})();
