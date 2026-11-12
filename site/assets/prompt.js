// The homepage's two copy controls. The agents band's pill copies the prompt that installs
// Andrey's agent skill, to paste into a coding agent. Enhancement only: without this script, or
// where the clipboard refuses, the pill leads to the FAQ answer that gives the prompt.
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

// The hero's install line: the copy button, hidden without this script, copies the command the
// radio buttons show.
(() => {
  const button = document.querySelector(".install-copy");
  if (!button || !navigator.clipboard) return;
  const install = button.closest(".install");
  let reset;
  button.hidden = false;
  button.addEventListener("click", () => {
    const chosen = install.querySelector("input:checked").value;
    const line = install.querySelector(`.install-line--${chosen} code`).textContent;
    navigator.clipboard.writeText(line).then(() => {
      button.classList.add("is-copied");
      button.setAttribute("aria-label", "Copied");
      clearTimeout(reset);
      reset = setTimeout(() => {
        button.classList.remove("is-copied");
        button.setAttribute("aria-label", "Copy the command");
      }, 2000);
    });
  });
})();
