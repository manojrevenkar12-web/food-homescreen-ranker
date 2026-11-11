import { color, font, radius, size, space, weight } from "../theme/tokens";

export const buttonStyle = {
  font: `${weight.semibold} ${size.sm}px ${font.body}`,
  color: color.card,
  background: color.ink,
  border: "none",
  borderRadius: radius.pill,
  padding: `${space[2]}px ${space[5]}px`,
  cursor: "pointer",
} as const;

export function Skeleton({ label }: { label: string }) {
  const block = (w: number | string, h: number, r: number = radius.md) => ({ width: w, height: h, borderRadius: r, background: color.panel });
  return (
    <div role="status" aria-busy="true" aria-label={label} style={{ display: "flex", gap: space[6], flexWrap: "wrap", padding: `${space[6]}px 0` }}>
      {[0, 1].map((p) => (
        <div key={p} style={{ ...block(300, 520, radius.phone), padding: space[5], display: "grid", gap: space[4], alignContent: "start" }}>
          {[0, 1, 2].map((r) => (
            <div key={r} style={{ display: "flex", gap: space[2] }}>
              {[0, 1].map((c) => (
                <div key={c} style={{ ...block(112, 92), background: color.surface }} />
              ))}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export function ErrorNotice({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div role="alert" style={{ background: color.chiliSoft, borderRadius: radius.md, padding: `${space[4]}px ${space[5]}px`, margin: `${space[6]}px 0` }}>
      <p style={{ margin: 0, fontWeight: weight.semibold }}>The results did not load: {message}.</p>
      <p style={{ margin: `${space[2]}px 0 ${space[4]}px`, color: color.muted }}>
        <code>web/public/data.json</code> is written by <code>python scripts/report.py</code>. Check that it exists, then load it again.
      </p>
      <button type="button" onClick={onRetry} style={buttonStyle}>
        Load again
      </button>
    </div>
  );
}

export function EmptyNotice({ what, command }: { what: string; command: string }) {
  return (
    <p style={{ margin: 0, padding: space[4], borderRadius: radius.md, background: color.panel, color: color.muted }}>
      No {what} yet. Run <code>{command}</code> to produce them.
    </p>
  );
}
