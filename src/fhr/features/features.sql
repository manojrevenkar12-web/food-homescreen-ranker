-- Ranking features for every (session, merchant) candidate, point in time.
-- Runs unchanged on DuckDB (development, tests) and Spark SQL (the daily Airflow job): plain SQL only.
--   inputs: candidates(sid, merchant, distance, eta, load)   serve-time values, logged with the impression
--           sessions(sid, day, hour, customer)
--           orders(sid, customer, merchant, late, stars)
--           impressions(sid, merchant, clicked)
--           merchants(merchant, cuisine, price, promoted, commission, joined_day, breakfast)
-- Customer features read every order before this session (streaming counters in production).
-- Merchant features read whole days before this session's day (the nightly batch).

WITH cand AS (
    SELECT c.sid, c.merchant, s.customer, s.day, s.hour, c.distance, c.eta, c.load
    FROM candidates c JOIN sessions s ON s.sid = c.sid
),
ord AS (
    SELECT o.sid, o.customer, o.merchant, s.day, CAST(o.late AS INT) AS late, o.stars, m.cuisine, m.price
    FROM orders o JOIN sessions s ON s.sid = o.sid JOIN merchants m ON m.merchant = o.merchant
),
cm AS (
    SELECT c.sid, c.merchant, COUNT(o.sid) AS cm_orders, MAX(o.day) AS cm_last_day
    FROM cand c LEFT JOIN ord o ON o.customer = c.customer AND o.merchant = c.merchant AND o.sid < c.sid
    GROUP BY c.sid, c.merchant
),
cust AS (
    SELECT c.sid, c.merchant, COUNT(o.sid) AS c_orders,
           SUM(CASE WHEN o.cuisine = m.cuisine THEN 1 ELSE 0 END) AS c_same_cuisine,
           AVG(o.price) AS c_price_avg
    FROM cand c JOIN merchants m ON m.merchant = c.merchant
    LEFT JOIN ord o ON o.customer = c.customer AND o.sid < c.sid
    GROUP BY c.sid, c.merchant
),
mday AS (
    SELECT merchant, day, COUNT(*) AS n, SUM(late) AS late, SUM(stars) AS stars_sum, COUNT(stars) AS stars_n
    FROM ord GROUP BY merchant, day
),
iday AS (
    SELECT i.merchant, s.day, COUNT(*) AS impr, SUM(CAST(i.clicked AS INT)) AS clicks
    FROM impressions i JOIN sessions s ON s.sid = i.sid GROUP BY i.merchant, s.day
),
mer AS (
    SELECT c.sid, c.merchant,
           SUM(CASE WHEN d.day >= c.day - 7 THEN d.n ELSE 0 END) AS m_orders_7d,
           SUM(d.n) AS m_orders, SUM(d.late) AS m_late, SUM(d.stars_sum) AS m_stars_sum, SUM(d.stars_n) AS m_stars_n
    FROM cand c LEFT JOIN mday d ON d.merchant = c.merchant AND d.day < c.day
    GROUP BY c.sid, c.merchant
),
exp AS (
    SELECT c.sid, c.merchant, SUM(d.impr) AS m_impr, SUM(d.clicks) AS m_clicks
    FROM cand c LEFT JOIN iday d ON d.merchant = c.merchant AND d.day < c.day
    GROUP BY c.sid, c.merchant
)
SELECT c.sid, c.merchant, c.customer, c.day, CAST(c.hour AS DOUBLE) AS hour,
       CAST(COALESCE(cm.cm_orders, 0) AS DOUBLE) AS cm_orders,
       CAST(CASE WHEN cm.cm_last_day IS NULL THEN 99 ELSE c.day - cm.cm_last_day END AS DOUBLE) AS cm_days_since,
       CAST(COALESCE(cu.c_orders, 0) AS DOUBLE) AS c_orders,
       CASE WHEN COALESCE(cu.c_orders, 0) = 0 THEN 0.0 ELSE cu.c_same_cuisine * 1.0 / cu.c_orders END AS c_cuisine_share,
       CASE WHEN COALESCE(cu.c_orders, 0) = 0 THEN 0.0 ELSE m.price - cu.c_price_avg END AS c_price_gap,
       CAST(COALESCE(me.m_orders_7d, 0) AS DOUBLE) AS m_orders_7d,
       CAST(COALESCE(me.m_orders, 0) AS DOUBLE) AS m_orders,
       CASE WHEN COALESCE(me.m_orders, 0) = 0 THEN 0.0 ELSE me.m_late * 1.0 / me.m_orders END AS m_late_rate,
       (COALESCE(me.m_stars_sum, 0) + 3.6 * 5) / (COALESCE(me.m_stars_n, 0) + 5) AS m_rating,
       (COALESCE(ex.m_clicks, 0) + 1.0) / (COALESCE(ex.m_impr, 0) + 10.0) AS m_ctr,
       CAST(c.day - m.joined_day AS DOUBLE) AS m_age_days,
       CAST(m.price AS DOUBLE) AS price, CAST(m.promoted AS DOUBLE) AS promoted, m.commission,
       CAST(m.cuisine AS DOUBLE) AS cuisine,
       CAST(CASE WHEN c.hour < 11 AND m.breakfast = 1 THEN 1 ELSE 0 END AS DOUBLE) AS morning_fit,
       c.distance, c.eta, c.load
FROM cand c
JOIN merchants m ON m.merchant = c.merchant
LEFT JOIN cm ON cm.sid = c.sid AND cm.merchant = c.merchant
LEFT JOIN cust cu ON cu.sid = c.sid AND cu.merchant = c.merchant
LEFT JOIN mer me ON me.sid = c.sid AND me.merchant = c.merchant
LEFT JOIN exp ex ON ex.sid = c.sid AND ex.merchant = c.merchant
ORDER BY c.sid, c.merchant
