"""Physical identity of qualified local artifact storage, without filesystem IO."""

from cayu.artifacts.local import LocalArtifactStore


def local_artifact_store_identity(store: LocalArtifactStore) -> dict[str, object]:
    return {
        "id": store.id,
        "root": str(store.root),
        "root_identity": [str(part) for part in store._root_identity],
    }
