"""Experimental seams for adapters maintained outside the Cayu distribution.

``cayu.extensions.runners`` and ``cayu.extensions.egress`` publish the shared
helpers Cayu's own sandbox adapters use, so that an independently installed
adapter does not need private Cayu imports. These modules are experimental:
names and signatures may change between releases until the supported extension
contract and compatibility tiers are finalized. See ``docs/build-a-runner.md``.
"""
