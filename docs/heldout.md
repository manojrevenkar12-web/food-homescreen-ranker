# The held-out city

`config/world_heldout.toml` is a city none of the design work looked at: other cuisine weights, 12% new
restaurants instead of 8%, kitchens with less capacity (2.6 orders an hour instead of 3.0), 520 merchants and
14,000 customers. Its config, the platform settings, the feature SQL, every model and both learned policies were
pinned by hash in `config/heldout.lock` before it ran.

`scripts/heldout.py` refuses to run if any pinned file changed or if `reports/heldout.json` already exists. It ran
once, on seed 47, with the lock intact.

| | Today's screen | Re-ranked rows | Assembled page |
|---|---|---|---|
| Orders per session | 0.573 | 0.636 (+11.0%) | 0.729 (+27.2%) |
| Late orders | 4.5% | 2.9% | 0.7% |
| Orders to new merchants | 15.1% | 13.6% | 12.8% |
| Orders on the top 10% of merchants | 66.3% | 63.3% | 50.8% |
| NDCG@8, LambdaRank vs the page as served | | | 0.579 vs 0.497 |

The lift holds up: +27% here against +26% to +31% in the three development cities. Late orders fall as they did
there, even with tighter kitchens. New merchants lose some share to the assembled page here too; with more new
merchants in the city, the page still gives them one order in eight.

After the run, the CUPED analysis in `src/fhr/evals/experiments.py` was corrected (see the README). That file is
not pinned and the held-out run does not use it.
