import { Designs } from "../features/experiments/Designs";
import { Gate } from "../features/gate/Gate";
import { Models } from "../features/models/Models";
import { Policies } from "../features/policies/Policies";
import { Screens } from "../features/screen/Screens";
import type { Data } from "../shared/data/types";
import { browserFetch, type Fetcher, useJson } from "../shared/data/useJson";
import { signed } from "../shared/format";
import { color, font, leading, measure, size, space, weight } from "../shared/theme/tokens";
import { Section } from "../shared/ui/Section";
import { ErrorNotice, Skeleton } from "../shared/ui/States";

export function App({ fetcher = browserFetch }: { fetcher?: Fetcher }) {
  const load = useJson<Data>("./data.json", fetcher);
  const page = load.state === "ready" ? load.data.policies.find((p) => p.name === "page") : undefined;
  return (
    <main style={{ maxWidth: measure.page, margin: "0 auto", padding: `${space[8]}px ${space[5]}px` }}>
      <header style={{ marginBottom: space[7] }}>
        <h1 style={{ margin: 0, fontFamily: font.display, fontWeight: weight.heavy, fontSize: size.display, lineHeight: leading.tight, letterSpacing: "-0.02em", maxWidth: "16ch" }}>
          The home screen, assembled
        </h1>
        <p style={{ margin: `${space[4]}px 0 0`, fontSize: size.lg, color: color.muted, maxWidth: `${measure.text}ch` }}>
          Which rows a customer sees, in what order, and what is in them: two-tower retrieval, LambdaRank, and a page built row by row
          for expected orders, with commission, new restaurants and busy kitchens weighed in.
          {page && <> In three simulated cities, {signed(page.lift)} orders per session over today's screen.</>}
        </p>
      </header>
      {load.state === "loading" && <Skeleton label="Loading the results" />}
      {load.state === "error" && <ErrorNotice message={load.message} onRetry={load.retry} />}
      {load.state === "ready" && (
        <>
          <Screens samples={load.data.samples} />
          <Section id="policies" title="Against the truth" lede="Every home screen ran on the same simulated week: same customers, same moods, same scrolling. Only the page differed.">
            <Policies rows={load.data.policies} />
          </Section>
          <Section id="designs" title="Which A/B test to trust" lede="Kitchens are shared, so a customer split lets each arm change the other's delivery times. Four replications of each design against the true effect.">
            <Designs experiments={load.data.experiments} />
          </Section>
          <Section id="models" title="The models underneath" lede="Retrieval, ranking and the order-again challenger, each scored on days the models never saw.">
            <Models t={load.data.training} />
          </Section>
          <Section id="gate" title="Ship or hold" lede="Decided from the first switchback and the serving benchmark; the truth is never read.">
            <Gate gate={load.data.gate} />
          </Section>
          <footer style={{ borderTop: `1px solid ${color.line}`, paddingTop: space[5], color: color.muted, fontSize: size.sm }}>
            Every number here comes from <code>reports/*.json</code>, collected by <code>python scripts/report.py</code>.
          </footer>
        </>
      )}
    </main>
  );
}
