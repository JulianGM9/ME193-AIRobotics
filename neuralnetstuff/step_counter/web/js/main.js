// App shell: one view at a time. Only the Label view exists so far (M3); the other tabs
// are placeholders for later milestones (PLAN.md D17).
import { mountLabel } from "./label.js";

const VIEWS = { label: mountLabel };

function show(name) {
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("active", b.dataset.view === name);
  const root = document.getElementById("view");
  root.replaceChildren();
  VIEWS[name](root);
}

document.getElementById("tabs").addEventListener("click", (e) => {
  const name = e.target.dataset?.view;
  if (name && VIEWS[name]) show(name);
});

show("label");
