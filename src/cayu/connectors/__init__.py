"""Ready tools for an application's connections, one toolkit per connector.

``cayu.connectors.slack`` and ``cayu.connectors.snowflake`` each publish ``tools(client=...,
slot=..., include_writes=False)``, the toolkit's read tools on one slot, and its write tools
too when asked; ``approval_policy(inner)``, a tool policy that asks
for approval on the toolkit's write tools and leaves every other decision to ``inner``; and
``review_fields()``, the arguments an approver sees, for ``ToolArgumentReview``. Every call goes
through a ``cayu.connections`` client: Cayu Cloud's broker by default, or ``LocalConnections``
for a laptop or a test. ``cayu.connectors.sap`` only reads: it publishes ``tools(client=...,
slot=...)`` alone, with nothing to approve. See ``docs/connector-toolkits.md``.
"""
