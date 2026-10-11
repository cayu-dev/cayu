"""Session-store cases organized by contract, with reusable test support.

The canonical suite is tests/core/test_session_store_shared_conformance.py.
Its explicit imports preserve collected IDs, order, markers and the shared
PostgreSQL schema lifetime. These modules are not collected a second time by
the default test-file pattern. Select a module explicitly for a focused run,
for example: pytest tests/core/session_store_conformance/publication.py.
"""
