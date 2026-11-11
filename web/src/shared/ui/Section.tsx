import type { ReactNode } from "react";
import { color, font, leading, measure, size, space, stroke, weight } from "../theme/tokens";

export function Section({ id, title, lede, children }: { id: string; title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section aria-labelledby={`${id}-h`} style={{ padding: `${space[8]}px 0`, borderTop: `${stroke.hair}px solid ${color.line}` }}>
      <h2 id={`${id}-h`} style={{ margin: 0, fontFamily: font.display, fontSize: size.xxl, fontWeight: weight.heavy, lineHeight: leading.tight, maxWidth: `${measure.text}ch` }}>
        {title}
      </h2>
      {lede && <p style={{ margin: `${space[3]}px 0 0`, color: color.muted, maxWidth: `${measure.text}ch` }}>{lede}</p>}
      <div style={{ marginTop: space[6] }}>{children}</div>
    </section>
  );
}
