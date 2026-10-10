"""Compaction operations and shared dispatch identity authority.

Explicit compaction owns a session-operation claim; automatic compaction owns
model-stage authority and context publication. Each entrance is composed from
stores, accounting and event publication without calling session orchestration.
"""
