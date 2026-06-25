"""
Recomputes the 10 most-active locations (pickups + dropoffs combined) from
the combined Yellow+FHVHV 2021 source, since the hardcoded list left over
from earlier tasks was based on Yellow Taxi alone and will shift once FHVHV
(much higher trip volume) is mixed in.
"""

import duckdb

from config import COMBINED_PARQUET

con = duckdb.connect()

result = con.sql(f"""
    WITH pu_counts AS (
        SELECT PULocationID AS LocationID, COUNT(*) AS n
        FROM read_parquet('{COMBINED_PARQUET}')
        GROUP BY PULocationID
    ),
    do_counts AS (
        SELECT DOLocationID AS LocationID, COUNT(*) AS n
        FROM read_parquet('{COMBINED_PARQUET}')
        GROUP BY DOLocationID
    )
    SELECT
        LocationID,
        SUM(n) AS total_pickups_and_dropoffs
    FROM (
        SELECT * FROM pu_counts
        UNION ALL
        SELECT * FROM do_counts
    )
    GROUP BY LocationID
    ORDER BY total_pickups_and_dropoffs DESC
    LIMIT 10
""").df()

print(result)
print()
print("TOP10_LOCATIONS =", result["LocationID"].tolist())