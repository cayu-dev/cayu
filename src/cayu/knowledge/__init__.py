"""Knowledge curation, enrichment, governance, and maintenance.

Import capabilities from their owning modules, for example::

    from cayu.knowledge.records import KnowledgeEntry, KnowledgeRevisionRef
    from cayu.knowledge.relations import KnowledgeRelation, KnowledgeLineageQuery
    from cayu.knowledge.maintenance_contracts import KnowledgeMaintenanceProposal
    from cayu.knowledge.activation_contracts import KnowledgeActivationRequest
    from cayu.knowledge.changes import KnowledgeChange, KnowledgeChangeBatch
    from cayu.knowledge.indexing import KnowledgeIndexReadiness, knowledge_chunk_embedding_identity
    from cayu.knowledge.search import KnowledgeQuery, KnowledgeSearchResult, KnowledgeListQuery
    from cayu.knowledge.publication_contracts import prepare_knowledge_publication
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.knowledge.curator import KnowledgeCurator
    from cayu.knowledge.governance import KnowledgeActivationPolicy

This package intentionally performs no eager capability re-exports.
"""
