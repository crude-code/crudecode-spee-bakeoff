-- AUDIT REFERENCE ONLY. Runtime extraction is implemented in extract_board_v2.py.
-- Frozen cutoff: 2024-06-01 (June 2024 production month)
-- Holdout: 2024-07-01 through 2025-06-01 (12 calendar months)
-- Missing calendar months must be reindexed explicitly; row counts are not elapsed time.
-- Formation-level play mapping; no Marcellus substitution and no Barnett claim.
WITH monthly AS (
  SELECT p.wellid,
         date_trunc('month', p.producingmonth)::date AS producingmonth,
         sum(p.liquidsprod_bbl)::float8 AS oil,
         sum(p.gasprod_mcf)::float8 AS gas,
         sum(p.waterprod_bbl)::float8 AS water
  FROM data.production p
  WHERE p.producingmonth <= '2025-06-01'
  GROUP BY p.wellid, date_trunc('month', p.producingmonth)::date
), play AS (
  SELECT w.wellid, w.api_uwi,
    CASE WHEN w.envbasin='DELAWARE' THEN 'DELAWARE'
         WHEN w.envbasin='MIDLAND' THEN 'MIDLAND'
         WHEN w.envinterval IN ('LOWER EAGLE FORD','UPPER EAGLE FORD') THEN 'EAGLE_FORD'
         WHEN w.envbasin='WILLISTON' THEN 'WILLISTON'
         WHEN w.envinterval IN ('NIOBRARA A','NIOBRARA B','NIOBRARA C','CODELL') THEN 'DJ'
         WHEN w.envinterval='HAYNESVILLE' THEN 'HAYNESVILLE' END AS play,
    w.envbasin, w.envinterval, w.laterallength_ft, w.firstproddate
  FROM data.wells w
  WHERE w.laterallength_ft >= 3000
    AND w.firstproddate BETWEEN '2020-07-01' AND '2023-12-31'
), agg AS (
  SELECT wellid,
    min(producingmonth) FILTER (
      WHERE producingmonth <= '2024-06-01'
        AND (coalesce(oil,0)>0 OR coalesce(gas,0)>0 OR coalesce(water,0)>0)
    ) AS first_prod_month,
    count(*) FILTER (WHERE producingmonth <= '2024-06-01') AS train_reported_mo,
    count(*) FILTER (WHERE producingmonth > '2024-06-01' AND producingmonth <= '2025-06-01') AS hold_reported_mo,
    sum(coalesce(oil,0)) FILTER (WHERE producingmonth <= '2024-06-01') AS oil_tr,
    sum(coalesce(gas,0)) FILTER (WHERE producingmonth <= '2024-06-01') AS gas_tr
  FROM monthly GROUP BY wellid
)
SELECT p.*, a.*,
  (extract(year from age('2024-06-01'::date, a.first_prod_month))::int * 12
   + extract(month from age('2024-06-01'::date, a.first_prod_month))::int + 1) AS train_span_mo,
  md5(p.wellid::text || 'seed20260728-v2') AS ord_hash
FROM play p JOIN agg a USING (wellid)
WHERE p.play IS NOT NULL
  AND a.first_prod_month IS NOT NULL
  AND a.hold_reported_mo >= 11
  AND (coalesce(a.oil_tr,0)>0 OR coalesce(a.gas_tr,0)>0)
  AND (extract(year from age('2024-06-01'::date, a.first_prod_month))::int * 12
       + extract(month from age('2024-06-01'::date, a.first_prod_month))::int + 1) BETWEEN 7 AND 48
ORDER BY p.play, ord_hash;
