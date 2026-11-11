export const pct = (x: number, d = 1) => `${(100 * x).toFixed(d)}%`;
export const signed = (x: number, d = 1) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(100 * x).toFixed(d)}%`;
export const pp = (x: number, d = 2) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(100 * x).toFixed(d)} pp`;
