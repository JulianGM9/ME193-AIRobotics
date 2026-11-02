// App shell: one view at a time. Label (M3) and Live (M7); Datasets, Train and Models were
// cut for the deadline (PLAN.md), so their tabs stay disabled.
import { mountLabel } from "./label.js";
import { mountLive } from "./live.js";

const VIEWS = { label: mountLabel, live: mountLive };

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

// #live or #live=S005 (replay S005 straight away) opens the Live view; anything else Label
show(location.hash.startsWith("#live") ? "live" : "label");
