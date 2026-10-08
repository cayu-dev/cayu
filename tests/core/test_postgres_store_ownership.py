"""Supported PostgreSQL class imports retain one identity after relocation."""

import importlib
import pickle

import pytest

import cayu


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("event_watchers_postgres", "PostgresEventWatcherStore")],
)
def test_postgres_owner_preserves_public_imports_and_pickled_class(module_name, class_name):
    owner = importlib.import_module("cayu.storage." + module_name)
    store_type = getattr(owner, class_name)
    public_stores = importlib.import_module("cayu.storage.postgres")
    assert getattr(public_stores, class_name) is store_type
    assert getattr(cayu, class_name) is store_type
    assert class_name not in cayu.__all__
    storage = importlib.import_module("cayu.storage")
    assert getattr(storage, class_name) is store_type
    assert class_name not in storage.__all__
    assert pickle.loads(pickle.dumps(store_type)) is store_type
    historical_reference = f"ccayu.storage.postgres\n{class_name}\n.".encode()
    assert pickle.loads(historical_reference) is store_type
