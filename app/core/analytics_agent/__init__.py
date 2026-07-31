"""Analytics Agent — natural-language questions over ally-be's analytics tables.

Two stateless transforms, no database access (ally-be owns Postgres and runs the
SQL itself):

- ``plan_query``   question + schema catalogue -> a single read-only SELECT
- ``compose_answer`` question + result rows    -> prose + a chart specification
"""
