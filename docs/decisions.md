# Decisions

Each entry: what was decided, what was rejected, and why.

**Log what was on screen, with the values serving used.** The app logs a card only when it was visible, together
with the distance, ETA estimate and kitchen load the page was built with. Training on rendered-but-unseen cards
would teach the ranker that cards below the fold are bad; recomputing ETA and load later would train on values
serving never had.

**One feature definition, three engines.** `features.sql` runs on DuckDB in development and on Spark in the
nightly job; `online()` computes the same values at serving time. Tests hold all three equal. The Spark test
caught a real difference on the first run: pandas marks a missing rating as NaN, and Spark kept it as a number
while DuckDB read it as NULL. Missing values now go to Spark as NULL with an explicit schema.

**Customer counters are live, merchant counters are nightly.** A customer's own orders change what they want
within minutes (order again, novelty), so they are read up to the session. Merchant popularity, ratings and
click rates come from whole days before today, as a nightly batch would provide them; the SQL and the serving
path both say so explicitly.

**The two-tower that makes the ranker's feature is trained on earlier days.** Otherwise the `tt_score` feature
would have seen the very orders the ranker learns from, and its weight would collapse at serving time.

**logQ correction in the two-tower.** Popular merchants show up as in-batch negatives more often; without the
correction the model learns that popular means unwanted.

**A sequential model ships only if it beats counts by two points of recall@10.** A GRU over the order sequence
is cheap here, but every model in the serving path is a thing to monitor. It passed in every city.

**Whole-page greedy assembly instead of a fixed template.** The value of a row depends on what is already above
it: a "for you" row and an "order again" row overlap for a regular customer. Greedy placement by marginal value,
with dedup across rows, handles that; the template cannot.

**A kitchen-load discount in the page value.** It barely moves orders, and cuts late deliveries by about two thirds,
because it stops the page from sending the evening rush to the kitchens that are already full.

**Commission in the objective, with a small weight.** It trades under one percent of orders for two to five percent
more commission per session. The weight is in config and the trade-off is reported per city.

**The exploration bonus stays, at a small weight.** In this city the dedicated new-restaurant row
and the guardrail already give new merchants their exposure; the bonus adds little on top: it moves orders by
less than a point either way and leaves the new-merchant share where it was. It stays because it costs little, and it is
the first knob to test again with real data, where nothing else may be giving new restaurants a place.

**A switchback, not a customer split, decides the release.** Customers in a split share kitchens, so each arm
changes the other's delivery times. The switchback gives the whole city one policy per block and drops the first
hour of each block, while the previous block's orders still load the kitchens. Both designs turned out to read
the effect low (see the README); the gate uses the switchback's lower bound to decide ship or hold, not its
point estimate to forecast the launch.
