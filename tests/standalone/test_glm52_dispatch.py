# SPDX-License-Identifier: Apache-2.0
"""Preserve runtime KV topology when constructing the native GLM-5.2 adapter."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_checkpoint_wrapper import wrapper_class


def test_preimported_dynamic_wrapper_reaches_ascend_with_runtime_topology(monkeypatch):
    reached = []

    class BeforeResources(RuntimeError):
        pass

    class NativeImpl:
        def __init__(self, config, role, parent, kv_cache_config=None):
            reached.append((type(self), kv_cache_config, type(parent)))
            raise BeforeResources()

    adapter = ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
    adapter.LMCacheConnectorV1Impl = NativeImpl
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    # The production class is now native, not a plugin subclass that patches it.
    wrapper = wrapper_class(include_init=True)
    config = SimpleNamespace(kv_transfer_config=None, parallel_config=None)
    topology = SimpleNamespace(kv_cache_groups=[object(), object()])
    with pytest.raises(BeforeResources):
        wrapper(config, object(), topology)
    assert reached == [(NativeImpl, topology, wrapper)]

def test_graph_entry_forwards_to_the_ascend_impl_with_exact_request_order(monkeypatch):
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import LMCacheAscendConnectorV1Dynamic

    calls = []
    sources = (object(), None, object())

    def prepare(self, names, **kwargs):
        calls.append((type(self), names, kwargs))
        return sources

    monkeypatch.setattr(LMCacheAscendConnectorV1Impl, "prepare_sparse_graph_step", prepare)
    wrapper = object.__new__(LMCacheAscendConnectorV1Dynamic)
    wrapper._lmcache_engine = object.__new__(LMCacheAscendConnectorV1Impl)
    names, requests, frontiers = ("l0", "l1"), ("b", "empty", "a"), (4096, 0, 8192)
    assert wrapper.prepare_sparse_graph_step(names, request_ids=requests, frontiers=frontiers) is sources
    assert calls == [(LMCacheAscendConnectorV1Impl, names,
                      dict(allow_empty=False, request_ids=requests, frontiers=frontiers))]
