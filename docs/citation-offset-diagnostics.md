# Citation offset diagnostics

Invalid Responses URL-citation offsets retain the reason
`citation_has_invalid_text_offsets`. The additional
`provider_protocol_citation_*` fields record structural evidence only:

- `condition`: `missing_endpoint`, `non_integer_endpoint`, `negative_start`,
  `empty_range`, `reversed_range`, or `end_out_of_bounds`, in validation order.
- `start_kind` and `end_kind`: absent, null, boolean, integer, number, string,
  array, object, or non_json. No values of invalid types are retained.
- `start_index` and `end_index`: integers within ±(2³¹−1). Larger integers are
  omitted with the corresponding `*_index_status=outside_diagnostic_bounds`.
- `text_length` and `text_offset`: associated part length and its base in the
  assembled assistant text, in Python Unicode code points. Both are capped at
  2³¹−1; that value means at least that many code points.

No annotation text, title, URL, prompt, credentials, or opaque payload is added
by these diagnostics. The fields survive foreground errors, background error
sanitization, and durable model-error projection. Validation and retry policy
are unchanged. Ranges are never clamped or invented. The existing compatibility
case where both endpoints are absent/null still emits citation evidence without
positions.

The [Responses web-search documentation](https://developers.openai.com/api/docs/guides/tools-web-search#output-and-citations)
places annotations on an `output_text` part. Runtime currently interprets a
present range as a nonempty, half-open interval within that part using Python
code-point indices, then adds the part's assembled-text base. Synthetic tests
cover missing/null endpoints, booleans and other invalid types, oversized
integers, negative/empty/reversed/out-of-bounds ranges, multiple text parts,
non-ASCII BMP characters, astral characters, combining marks, completion,
usage, and retry exhaustion.

The documentation does not resolve all Unicode indexing details. These tests
establish Runtime's existing interpretation, not upstream Unicode equivalence.
No original failing annotation or isolated upstream/intermediary capture was
available for #1601. The historical failure remains unattributed; changing
index units or text association requires sanitized evidence from that boundary.
No active workload or live provider request is needed for these controls.

A separate pre-existing stream limit remains: a message supplied only inside
`response.completed.output`, without text/annotation or output-item events,
is retained as provider state but its citations are not emitted/validated by
the foreground stream parser. Non-stream response parsing and incremental
annotation parsing are covered here; this change does not establish terminal-only
stream annotation parity.
