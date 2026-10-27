"""Shared SQL keyset plans for supported persistent eval read stores."""

from __future__ import annotations

from cayu.evals._report_publication import report_association_key
from cayu.evals.read_views import EvalViewQuery
from cayu.evals.reported import EvalReportedSeriesSelector


def view_index_sql(query: EvalViewQuery, *, postgres: bool) -> tuple[str, tuple[object, ...]]:
    """SQL keyset query shared by the two persistent stores, without route scans."""
    query = EvalViewQuery.model_validate(query)
    boundary = query.boundary()
    params: list[object] = []
    if query.suite is None:
        if query.target_key is not None:
            predicate, target_params = "target_key = ?", (query.target_key,)
        elif query.target_keys is not None:
            predicate = (
                "target_key IN (" + ",".join("?" for _ in query.target_keys) + ")"
                if query.target_keys
                else "1 = 0"
            )
            target_params = query.target_keys
        else:
            predicate, target_params = "1 = 1", ()
        union = f"""
            SELECT 'authored' AS source, revision, NULL AS suite_id, created_at, target_key,
                   'authored:' || revision || ':' AS sort_key
            FROM cayu_eval_authored_suites WHERE {predicate}
            UNION ALL
            SELECT 'corpus', suites.corpus_revision, suites.suite_id, corpora.created_at, corpora.target_key,
                   'corpus:' || suites.corpus_revision || ':' || suites.suite_id
            FROM cayu_eval_suites AS suites
            JOIN cayu_eval_corpora AS corpora ON corpora.revision = suites.corpus_revision
            WHERE {predicate.replace("target_key", "corpora.target_key")}
        """
        union += f""" UNION ALL
            SELECT 'reported_suite', selector_json, NULL, created_at, target_key,
                   'reported_suite:' || selector_json || ':'
            FROM cayu_eval_report_series WHERE {predicate}
        """
        params.extend((*target_params, *target_params, *target_params))
    elif isinstance(query.suite, EvalReportedSeriesSelector):
        union = """SELECT 'report' AS source, report_key AS revision, NULL AS suite_id,
                    created_at, target_key, 'report:' || report_key || ':' AS sort_key
                   FROM cayu_eval_reports WHERE series_id = ?"""
        params.append(query.suite.series_id)
        if query.target_key is not None:
            union += " AND target_key = ?"
            params.append(query.target_key)
        if query.target_keys is not None:
            if query.target_keys:
                union += " AND target_key IN (" + ",".join("?" for _ in query.target_keys) + ")"
                params.extend(query.target_keys)
            else:
                union += " AND 1 = 0"
    else:
        scope = query.suite
        if scope.source == "authored":
            expression = (
                "invocation_json::jsonb ->> 'authored_suite_revision'"
                if postgres
                else "json_extract(invocation_json, '$.authored_suite_revision')"
            )
            predicate = f"target_key = ? AND {expression} = ?"
            params.extend((query.target_key, scope.revision))
        else:
            predicate = "target_key = ? AND corpus_revision = ? AND suite_id = ?"
            params.extend((query.target_key, scope.revision, scope.suite_id))
        # UNION deduplicates by projected identity before the outer keyset sort.
        # A linguistic default must not collapse distinct run IDs such as Zed/zed.
        identity_collation = '"C"' if postgres else "BINARY"
        union = f"""SELECT 'run' AS source, run_id COLLATE {identity_collation} AS revision, NULL AS suite_id, created_at, target_key,
                    'run:' || run_id || ':' AS sort_key FROM cayu_eval_runs WHERE {predicate}"""
        if scope.source == "corpus":
            union += """ UNION ALL
                SELECT 'result', records.revision, NULL, records.created_at, records.target_key,
                       'result:' || records.revision || ':'
                FROM cayu_eval_result_records AS records
                WHERE records.target_key = ? AND records.corpus_revision = ? AND records.suite_id = ?
                  AND NOT EXISTS (SELECT 1 FROM cayu_eval_runs AS runs
                                  WHERE runs.result_revision = records.revision)
            """
            params.extend((query.target_key, scope.revision, scope.suite_id))
        # An explicit atomic addition admits captured execution into destination
        # history. UNION deduplicates the same immutable result across receipts.
        union += """ UNION
            SELECT 'result', records.revision, NULL, records.created_at, records.target_key,
                   'result:' || records.revision || ':'
            FROM cayu_eval_session_additions AS additions
            JOIN cayu_eval_result_records AS records ON records.revision = additions.result_revision
            WHERE additions.destination_selector = ? AND records.target_key = ?
              AND NOT EXISTS (SELECT 1 FROM cayu_eval_runs AS runs
                              WHERE runs.result_revision = records.revision)
        """
        params.extend((scope.id, scope.target_key))
        if scope.source == "corpus":
            union += """ UNION ALL SELECT 'report', report_key, NULL, created_at, target_key,
                         'report:' || report_key || ':' FROM cayu_eval_reports
                         WHERE association_key = ? AND target_key = ?"""
            params.extend((report_association_key(scope.id), scope.target_key))

    # Cursor validation uses Python lexical order, independent of DB locale.
    sort_key = 'sort_key COLLATE "C"' if postgres else "sort_key"
    where = ""
    if boundary is not None:
        if query.suite is None:
            where = f"WHERE {sort_key} > ?"
            params.append(boundary[1])
        else:
            where = f"WHERE created_at < ? OR (created_at = ? AND {sort_key} > ?)"
            observed_at = boundary[0]
            if observed_at is None:
                raise ValueError("History cursor requires a timestamp.")
            timestamp = observed_at if postgres else observed_at.isoformat()
            params.extend((timestamp, timestamp, boundary[1]))
    order = f"created_at DESC, {sort_key} ASC" if query.suite else f"{sort_key} ASC"
    sql = f"SELECT source, revision, suite_id, created_at, target_key FROM ({union}) AS views {where} ORDER BY {order} LIMIT ?"
    params.append(query.limit + 1)
    return (sql.replace("?", "%s") if postgres else sql), tuple(params)


def latest_case_results_sql(query, *, postgres: bool) -> tuple[str, tuple[object, ...]]:
    """One latest immutable result per requested case; no history-page iteration."""
    scope = query.suite
    ids = ", ".join("?" for _ in query.case_ids)
    params: list[object] = [scope.target_key, scope.revision]
    if scope.source == "authored":
        expression = (
            "runs.invocation_json::jsonb ->> 'authored_suite_revision'"
            if postgres
            else "json_extract(runs.invocation_json, '$.authored_suite_revision')"
        )
        predicate = f"runs.target_key = ? AND {expression} = ?"
    else:
        predicate = "runs.target_key = ? AND runs.corpus_revision = ? AND runs.suite_id = ?"
        params.append(scope.suite_id)
    params.extend(query.case_ids)
    candidates = f"""
        SELECT cases.case_id, 'run' AS source, runs.run_id AS revision, runs.created_at
        FROM cayu_eval_runs AS runs JOIN cayu_eval_cases AS cases
          ON cases.corpus_revision = runs.corpus_revision AND cases.suite_id = runs.suite_id
        WHERE {predicate} AND runs.result_revision IS NOT NULL AND cases.case_id IN ({ids})
    """
    if scope.source == "corpus":
        captured_case = (
            "captured_result::jsonb -> 'score' ->> 'case_id'"
            if postgres
            else "json_extract(captured_result_json, '$.score.case_id')"
        )
        candidates += f""" UNION ALL
            SELECT {captured_case}, 'result', revision, created_at
            FROM cayu_eval_result_records
            WHERE target_key = ? AND corpus_revision = ? AND suite_id = ?
              AND origin = 'captured_session' AND {captured_case} IN ({ids})
        """
        params.extend((scope.target_key, scope.revision, scope.suite_id, *query.case_ids))
    collation = ' COLLATE "C"' if postgres else ""
    sql = f"""
        SELECT case_id, source, revision FROM (
            SELECT candidates.*, ROW_NUMBER() OVER (
                PARTITION BY case_id{collation}
                ORDER BY created_at DESC, source{collation} ASC, revision{collation} ASC
            ) AS position FROM ({candidates}) AS candidates
        ) AS latest WHERE position = 1 ORDER BY case_id{collation} ASC
    """
    return (sql.replace("?", "%s") if postgres else sql), tuple(params)
