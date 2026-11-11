import { color, font } from "./tokens";

export function applyTheme(root: HTMLElement = document.documentElement) {
  root.style.setProperty("--surface", color.surface);
  root.style.setProperty("--ink", color.ink);
  root.style.setProperty("--focus", color.focus);
  root.style.setProperty("--font-body", font.body);
}
