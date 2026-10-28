# SPDX-License-Identifier: Apache-2.0
"""Unit and regression tests for the MLA+DSA two-group separate storage design
(NPU shared side).

Covers the NPU-side changes:
- CacheEngineKey / LayerCacheEngineKey kv_group field (hash, eq, to/from_string,
  to_dict/from_dict, with_new_worker_id, split_layers)
- token_database.process_tokens kv_group propagation
- config knobs (save_full_chunk_in_decode, dsa_two_groups)
- DSA index first-rank storage policy follows save_only_first_rank
- MemoryFormat new tags (KV_MLA_LATENT_FMT, KV_DSA_INDEX_FMT)
- Regression: existing key serialization and config defaults unchanged

Format detection tests (KVCacheFormat) are in the Ascend test file since
KVCacheFormat is defined in the lmcache_ascend package.
"""
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import MAX_KEY_LENGTH
from lmcache.v1.token_database import (
    DSA_INDEX_CACHE_SCHEMA,
    DSA_INDEX_CACHE_SCHEMA_TAG,
    ChunkedTokenDatabase,
)

# Local
from .utils import dumb_metadata, generate_tokens
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryFormat, TensorMemoryAllocator


# ---------------------------------------------------------------------------
# CacheEngineKey / LayerCacheEngineKey kv_group tests
# ---------------------------------------------------------------------------

class TestCacheEngineKeyKvGroup:

    def test_default_kv_group_is_zero(self):
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=42, dtype=torch.bfloat16,
        )
        assert key.kv_group == 0

    def test_kv_group_set_explicitly(self):
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=42, dtype=torch.bfloat16, kv_group=1,
        )
        assert key.kv_group == 1

    def test_keys_with_different_kv_group_are_not_equal(self):
        k0 = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        k1 = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        assert k0 != k1
        assert hash(k0) != hash(k1)

    def test_keys_with_same_kv_group_are_equal(self):
        k0a = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        k0b = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        assert k0a == k0b
        assert hash(k0a) == hash(k0b)

    def test_to_string_includes_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        s = key.to_string()
        parts = s.split("@")
        # model@ws@wid@hash@dtype@kv_group
        assert parts[5] == "1"

    def test_from_string_parses_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        s = key.to_string()
        parsed = CacheEngineKey.from_string(s)
        assert parsed.kv_group == 1
        assert parsed == key

    def test_from_string_backward_compat_no_kv_group(self):
        """Old keys without kv_group should parse with kv_group=0."""
        # Simulate old format: model@ws@wid@hash@dtype (no kv_group)
        old_str = "test@1@0@2a@bfloat16"
        parsed = CacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.model_name == "test"
        assert parsed.world_size == 1

    def test_from_string_backward_compat_with_tags(self):
        """Old keys with tags but no kv_group should parse correctly."""
        old_str = "test@1@0@2a@bfloat16@tag1%val1@tag2%val2"
        parsed = CacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.tags == (("tag1", "val1"), ("tag2", "val2"))

    def test_to_from_string_roundtrip_with_tags_and_kv_group(self):
        key = CacheEngineKey(
            "test", 2, 1, 99, torch.bfloat16, kv_group=1,
            request_configs={"lmcache.tag.foo": "bar"},
        )
        s = key.to_string()
        parsed = CacheEngineKey.from_string(s)
        assert parsed == key
        assert parsed.kv_group == 1
        assert parsed.tags == (("foo", "bar"),)

    def test_to_dict_includes_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        d = key.to_dict()
        assert d["kv_group"] == 1

    def test_from_dict_parses_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        d = key.to_dict()
        restored = CacheEngineKey.from_dict(d)
        assert restored == key
        assert restored.kv_group == 1

    def test_from_dict_default_kv_group_zero(self):
        """Old dicts without kv_group should default to 0."""
        d = {
            "__type__": "CacheEngineKey",
            "model_name": "test",
            "world_size": 1,
            "worker_id": 0,
            "chunk_hash": 42,
            "dtype": "bfloat16",
        }
        restored = CacheEngineKey.from_dict(d)
        assert restored.kv_group == 0

    def test_with_new_worker_id_preserves_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        new_key = key.with_new_worker_id(2)
        assert new_key.kv_group == 1
        assert new_key.worker_id == 2


class TestLayerCacheEngineKeyKvGroup:

    def test_layer_key_kv_group(self):
        key = LayerCacheEngineKey(
            "test", 1, 0, 42, torch.bfloat16, layer_id=5, kv_group=1,
        )
        assert key.kv_group == 1
        assert key.layer_id == 5

    def test_layer_keys_different_kv_group_not_equal(self):
        k0 = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=0)
        k1 = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        assert k0 != k1
        assert hash(k0) != hash(k1)

    def test_layer_key_to_string_includes_kv_group_and_layer(self):
        key = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        s = key.to_string()
        parts = s.split("@")
        # model@ws@wid@hash@dtype@kv_group@layer_id
        assert parts[5] == "1"
        assert parts[6] == "3"

    def test_layer_key_from_string_new_format(self):
        key = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        s = key.to_string()
        parsed = LayerCacheEngineKey.from_string(s)
        assert parsed.kv_group == 1
        assert parsed.layer_id == 3
        assert parsed == key

    def test_layer_key_from_string_old_format_no_kv_group(self):
        """Old format: model@ws@wid@hash@dtype@layer_id (no kv_group)."""
        old_str = "test@1@0@2a@bfloat16@5"
        parsed = LayerCacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.layer_id == 5

    def test_layer_key_from_string_old_format_with_tags(self):
        """Old format: model@ws@wid@hash@dtype@layer_id@tags."""
        old_str = "test@1@0@2a@bfloat16@5@foo%bar"
        parsed = LayerCacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.layer_id == 5
        assert parsed.tags == (("foo", "bar"),)

    def test_split_layers_preserves_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        layer_keys = key.split_layers(4)
        assert len(layer_keys) == 4
        for lk in layer_keys:
            assert lk.kv_group == 1
            assert isinstance(lk, LayerCacheEngineKey)


# ---------------------------------------------------------------------------
# Config knob tests
# ---------------------------------------------------------------------------

class TestConfigKnobs:

    def test_save_full_chunk_in_decode_default_false(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_full_chunk_in_decode is False

    def test_save_full_chunk_in_decode_from_env(self):
        os.environ["LMCACHE_SAVE_FULL_CHUNK_IN_DECODE"] = "true"
        try:
            config = LMCacheEngineConfig.from_env()
            assert config.save_full_chunk_in_decode is True
        finally:
            del os.environ["LMCACHE_SAVE_FULL_CHUNK_IN_DECODE"]

    def test_dsa_indexer_policy_follows_save_only_first_rank(self):
        metadata = dumb_metadata()
        metadata.use_mla = True

        config = LMCacheEngineConfig.from_defaults(
            extra_config={"save_only_first_rank": True}
        )
        assert LMCacheEngine._resolve_save_indexer_only_first_rank(
            config,
            metadata,
            save_only_first_rank=True,
            dsa_two_groups=True,
        ) is True

        legacy_split_config = LMCacheEngineConfig.from_defaults(
            extra_config={
                "save_only_first_rank": False,
                "save_indexer_only_first_rank": True,
            }
        )
        assert LMCacheEngine._resolve_save_indexer_only_first_rank(
            legacy_split_config,
            metadata,
            save_only_first_rank=False,
            dsa_two_groups=True,
        ) is False

    def test_dsa_two_groups_default_false(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.dsa_two_groups is False

    def test_dsa_two_groups_from_env(self):
        os.environ["LMCACHE_DSA_TWO_GROUPS"] = "1"
        try:
            config = LMCacheEngineConfig.from_env()
            assert config.dsa_two_groups is True
        finally:
            del os.environ["LMCACHE_DSA_TWO_GROUPS"]

    # --- Regression: existing knobs unchanged ---

    def test_regression_save_decode_cache_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_decode_cache is False

    def test_regression_save_unfull_chunk_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_unfull_chunk is False

    def test_regression_use_layerwise_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.use_layerwise is False


# ---------------------------------------------------------------------------
# Token database kv_group tests
# ---------------------------------------------------------------------------

class TestTokenDatabaseKvGroup:

    def test_process_tokens_default_kv_group_zero(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens))
        assert len(results) > 0
        for _, _, key in results:
            assert key.kv_group == 0

    def test_process_tokens_kv_group_one(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results) > 0
        for _, _, key in results:
            assert key.kv_group == 1

    def test_process_tokens_different_kv_groups_produce_different_keys(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results_0 = list(db.process_tokens(tokens=tokens, kv_group=0))
        results_1 = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results_0) == len(results_1)
        for (_, _, k0), (_, _, k1) in zip(results_0, results_1):
            assert k0 != k1
            assert k0.kv_group == 0
            assert k1.kv_group == 1

    def test_split_layers_with_kv_group(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata(kv_shape=(4, 2, 64, 8, 128))
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results) == 1
        _, _, key = results[0]
        layer_keys = key.split_layers(4)
        assert len(layer_keys) == 4
        for lk in layer_keys:
            assert lk.kv_group == 1

    def test_dsa_index_group_uses_internal_schema_tag(self):
        cfg = LMCacheEngineConfig.from_legacy(
            chunk_size=64,
            backend="cpu",
            dsa_two_groups=True,
        )
        metadata = dumb_metadata()
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)

        key0 = list(db.process_tokens(tokens=tokens, kv_group=0))[0][2]
        key1 = list(db.process_tokens(tokens=tokens, kv_group=1))[0][2]

        assert DSA_INDEX_CACHE_SCHEMA_TAG not in (key0.request_configs or {})
        assert key1.request_configs[DSA_INDEX_CACHE_SCHEMA_TAG] == (
            DSA_INDEX_CACHE_SCHEMA
        )
        assert key0 != key1

    def test_dsa_index_group_accepts_single_dtype_metadata(self):
        cfg = LMCacheEngineConfig.from_legacy(
            chunk_size=64,
            backend="cpu",
            dsa_two_groups=True,
        )
        metadata = dumb_metadata()
        assert len(metadata.get_dtypes()) == 1
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)

        _, _, key = list(db.process_tokens(tokens=tokens, kv_group=1))[0]

        assert key.dtype == metadata.kv_dtype
        assert key.request_configs[DSA_INDEX_CACHE_SCHEMA_TAG] == (
            DSA_INDEX_CACHE_SCHEMA
        )

    def test_dsa_index_schema_tag_preserves_user_tags(self):
        cfg = LMCacheEngineConfig.from_legacy(
            chunk_size=64,
            backend="cpu",
            dsa_two_groups=True,
        )
        metadata = dumb_metadata()
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)

        _, _, key = list(
            db.process_tokens(
                tokens=tokens,
                kv_group=1,
                request_configs={"lmcache.tag.user": "alice"},
            )
        )[0]

        assert key.request_configs["lmcache.tag.user"] == "alice"
        assert key.request_configs[DSA_INDEX_CACHE_SCHEMA_TAG] == (
            DSA_INDEX_CACHE_SCHEMA
        )

    def test_dsa_index_schema_tag_keeps_glm_path_keys_under_protocol_limit(self):
        cfg = LMCacheEngineConfig.from_legacy(
            chunk_size=64,
            backend="cpu",
            dsa_two_groups=True,
        )
        metadata = dumb_metadata()
        metadata.model_name = "/workspace/models/GLM-5.1-w4a8"
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)

        _, _, key = list(db.process_tokens(tokens=tokens, kv_group=1))[0]

        assert len(key.to_string()) <= MAX_KEY_LENGTH
        for layer_key in key.split_layers(96):
            assert len(layer_key.to_string()) <= MAX_KEY_LENGTH


# ---------------------------------------------------------------------------
# MemoryFormat tests
# ---------------------------------------------------------------------------

class TestMemoryFormat:

    def test_new_formats_exist(self):
        assert hasattr(MemoryFormat, "KV_MLA_LATENT_FMT")
        assert hasattr(MemoryFormat, "KV_DSA_INDEX_FMT")

    def test_mla_latent_is_alias_of_mla_fmt(self):
        assert MemoryFormat.KV_MLA_LATENT_FMT is MemoryFormat.KV_MLA_FMT

    def test_new_formats_have_token_dim(self):
        assert MemoryFormat.KV_MLA_LATENT_FMT.token_dim() == 2
        assert MemoryFormat.KV_DSA_INDEX_FMT.token_dim() == 2

    def test_regression_existing_formats_token_dim(self):
        assert MemoryFormat.KV_2LTD.token_dim() == 2
        assert MemoryFormat.KV_T2D.token_dim() == 1
        assert MemoryFormat.KV_MLA_FMT.token_dim() == 2

def test_failed_direct_preflight_uses_overlapped_layerwise_store(monkeypatch) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    calls = []
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "save_kv_layer",
        lambda _self, layer, *_args, **_kwargs: calls.append(layer),
    )
    request = SimpleNamespace(req_id="request")
    adapter = _ascend_adapter_fake(
        _direct_prefill_requests=lambda: [request],
        _preflight_direct_store=lambda _requests: False,
        _latent_layer_names=["layer.0"],
        _indexer_layer_names=[],
        config=SimpleNamespace(dsa_two_groups=False),
    )

    adapter.save_kv_layer("layer.0", None, None)

    assert calls == ["layer.0"]
    assert adapter._direct_store_step_supported is False

def test_group0_remote_fill_rebuild_skips_group1_metadata() -> None:
    class _TokenDatabase:
        def process_tokens(self, **kwargs):
            assert kwargs["kv_group"] == 0
            return [(0, 1024, SimpleNamespace(chunk_hash=91, kv_group=0))]

    engine = SimpleNamespace(
        token_database=_TokenDatabase(),
        _remote_fill_direct_groups=lambda: (0,),
    )

    plans = AscendLMCacheEngine._remote_fill_prefix_plans(
        engine,
        list(range(1024)),
        {"lmcache.remote_fill": {"transfer_id": "transfer"}},
        1024,
    )

    assert set(plans) == {0}

def test_remote_fill_rebuilds_probe_keys_for_full_prefix_hit() -> None:
    class _TokenDatabase:
        def process_tokens(self, **kwargs):
            group = kwargs["kv_group"]
            if group == 0:
                assert len(kwargs["tokens"]) == 1024
            else:
                assert kwargs["hashes"] == [91]
                assert kwargs["offsets"] == [1024]
            return [(0, 1024, SimpleNamespace(chunk_hash=91, kv_group=group))]

    engine = SimpleNamespace(
        token_database=_TokenDatabase(),
        _remote_fill_direct_groups=lambda: (0, 1),
    )
    plans = AscendLMCacheEngine._remote_fill_prefix_plans(
        engine,
        list(range(1024)),
        {"lmcache.remote_fill": {"transfer_id": "transfer"}},
        1024,
    )

    assert set(plans) == {0, 1}
    assert plans[0][0][:2] == (0, 1024)
    assert plans[1][0][:2] == (0, 1024)

def test_remote_fill_submission_is_owned_by_finish_batch(
    is_last_prefill: bool,
    expected_final: bool,
) -> None:
    calls = []
    engine = SimpleNamespace(
        discard_live_source_descriptor=lambda *_args: None,
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            (args, kwargs)
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(
            dsa_two_groups=True,
            remote_fill_submission_mode="per_chunk",
        ),
        _remote_store_requested=True,
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["latent"], 1: ["indexer"]},
            {0: "latent-slots", 1: "indexer-slots"},
            16384,
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=list(range(18878)),
        request_configs=_adapter_remote_fill_request_configs(),
        load_spec=None,
        live_source_requested=True,
        is_last_prefill=is_last_prefill,
        _lmcache_remote_fill_qualified=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter,
        [request],
    )
    assert calls == []
    event = object()
    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter,
        [request],
        finish_batch=True,
        source_ready_event=event,
        source_ready_event_source="forward_context.sfa_reshape_cache_event",
        source_ready_events=(event,),
    )

    assert len(calls) == 1
    assert calls[0][1]["final"] is expected_final
    assert calls[0][1]["source_ready_events"] == (event,)

def test_remote_fill_selects_both_authoritative_groups() -> None:
    adapter = _ascend_adapter_fake(_remote_store_requested=True)
    request = SimpleNamespace(
        request_configs=_adapter_remote_fill_request_configs(),
        save_spec=SimpleNamespace(
            can_save_latent=False,
            can_save_indexer=False,
        ),
    )
    groups = {0: ["latent"], 1: ["indexer"]}

    selected = _ascend_adapter_method("_direct_selected_groups")(
        adapter, request, groups
    )

    assert selected == groups

def test_remote_fill_disabled_stale_handoff_does_not_disable_live_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    calls = []
    engine = SimpleNamespace(
        discard_live_source_descriptor=lambda *_args: pytest.fail(
            "an unqualified remote-fill hint must not disable legacy live source"
        ),
        begin_live_source_descriptor=lambda *args: calls.append(("begin", args)),
        capture_live_source_step=lambda *args: calls.append(("capture", args)),
        finalize_live_source_descriptor=lambda *args: (
            calls.append(("finalize", args)) or True
        ),
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            ("store", args, kwargs)
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _remote_store_requested=False,
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["latent"], 1: ["indexer"]},
            {0: "latent-slots", 1: "indexer-slots"},
            0,
        ),
    )
    request_configs = _adapter_remote_fill_request_configs()
    request_configs["lmcache.mooncake_preferred_segment"] = "legacy-decoder"
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2, 3],
        slot_mapping=["latent-slots"],
        indexer_slot_mapping=["indexer-slots"],
        save_slot_mapping_base=0,
        save_spec=None,
        load_spec=None,
        request_configs=request_configs,
        live_source_requested=True,
        is_last_prefill=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter,
        [request],
        source_ready_event=object(),
    )

    assert [item[0] for item in calls] == [
        "begin",
        "capture",
        "finalize",
        "store",
    ]
    assert calls[-1][2]["final"] is True
    assert request_configs["lmcache.mooncake_preferred_segment"] == (
        "legacy-decoder"
    )

def test_remote_fill_disabled_preserves_legacy_group_selection() -> None:
    request_configs = _adapter_remote_fill_request_configs()
    request_configs["lmcache.remote_fill"]["transfer_id"] = "stale-transfer"
    request_configs["lmcache.mooncake_preferred_segment"] = (
        "legacy-decoder-host"
    )
    adapter = _ascend_adapter_fake(_remote_store_requested=False)
    request = SimpleNamespace(
        request_configs=request_configs,
        save_spec=SimpleNamespace(
            can_save_latent=False,
            can_save_indexer=False,
        ),
    )

    assert (
        request_configs["lmcache.mooncake_preferred_segment"]
        == "legacy-decoder-host"
    )
    assert _ascend_adapter_method("_direct_selected_groups")(
        adapter,
        request,
        {0: ["latent"], 1: ["indexer"]},
    ) == {}

def test_group1_direct_persistence_requires_decoder_segment() -> None:
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        _prepare_remote_fill_persistent_placement,
    )

    with pytest.raises(
        ValueError, match="requires a decoder-local Mooncake segment"
    ):
        _prepare_remote_fill_persistent_placement(
            _adapter_remote_fill_request_configs(), group1_direct_hbm=True
        )

def test_remote_fill_persistence_selects_group1_decoder_segment() -> None:
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        _prepare_remote_fill_persistent_placement,
    )
    request_configs = _adapter_remote_fill_request_configs()
    request_configs["lmcache.mooncake_preferred_segment"] = "decoder-host"

    assert _prepare_remote_fill_persistent_placement(
        request_configs, group1_direct_hbm=True
    ) is True
    assert (
        request_configs["lmcache.mooncake_preferred_segment"]
        == "decoder-host"
    )
    assert request_configs["lmcache.mooncake_preferred_kv_group"] == 1

    legacy = _adapter_remote_fill_request_configs()
    legacy["lmcache.mooncake_preferred_segment"] = "decoder-host"
    assert _prepare_remote_fill_persistent_placement(legacy) is True
    assert "lmcache.mooncake_preferred_segment" not in legacy
    assert "lmcache.mooncake_preferred_kv_group" not in legacy

    ordinary = {"lmcache.mooncake_preferred_segment": "decoder-host"}
    assert _prepare_remote_fill_persistent_placement(ordinary) is False
    assert ordinary["lmcache.mooncake_preferred_segment"] == "decoder-host"

def _adapter_remote_fill_request_configs() -> dict:
    return {
        "lmcache.remote_fill": {
            "transfer_id": "transfer",
            "request_attempt": 1,
            "source_engine_id": "prefiller",
            "destination_engine_id": "decoder",
            "destination_engine_epoch": 7,
            "control_endpoint": "tcp://decoder:19001",
            "destination_dp_rank": 0,
            "shared_cache_generation": 3,
            "destination_tp_size": 8,
            "destination_dp_size": 1,
            "global_te_push": True,
            "token_hash_algorithm": "sha256",
            "python_hash_seed": "",
        }
    }

def test_remote_fill_terminal_is_returned_without_private_capabilities(
    monkeypatch,
) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "request_finished",
        lambda *_args: (False, {"first_tok": 7}),
    )
    terminal = {
        "transfer_id": "transfer",
        "outcome": "LOCAL_FULL",
        "persistent_common_end": 4096,
        "required_store_end": 4096,
    }
    adapter = _ascend_adapter_fake(
        _scheduler_live_sources={},
        _scheduler_remote_fill_results={"request": terminal},
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                data_parallel_index=1,
            )
        ),
        store_async=True,
        kv_role="kv_producer",
    )
    request = SimpleNamespace(
        request_id="request",
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_engine_id": "decoder",
            "lmcache.remote_fill": {
                "transfer_id": "transfer",
                "request_attempt": 1,
                "source_engine_id": "prefiller",
                "destination_engine_id": "decoder",
                "destination_engine_epoch": 2,
                "control_endpoint": "tcp://private:19001",
                "destination_dp_rank": 1,
                "shared_cache_generation": 3,
                "destination_tp_size": 8,
                "destination_dp_size": 2,
                "global_te_push": True,
                "token_hash_algorithm": "sha256",
                "python_hash_seed": "",
                "capability_mac": "must-not-leak",
                "destination_ptr": 1234,
            },
        },
    )

    delay_free, returned = _ascend_adapter_method("request_finished")(
        adapter, request, []
    )

    assert delay_free is True
    assert returned["first_tok"] == 7
    assert returned["do_remote_decode"] is True
    assert returned["remote_engine_id"] == "decoder"
    public = returned["lmcache.remote_fill"]
    assert public["terminal"] == terminal
    assert "control_endpoint" not in public
    assert "capability_mac" not in public
    assert "destination_ptr" not in public

def test_live_source_is_published_to_following_connector(monkeypatch) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "request_finished",
        lambda *_args: (False, None),
    )
    descriptor = {"tp_rank": 0, "dp_rank": 0}
    adapter = _ascend_adapter_fake(
        _scheduler_live_sources={"request": [descriptor]},
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                data_parallel_rank_local=0,
            )
        ),
        store_async=True,
        kv_role="kv_producer",
    )
    request = SimpleNamespace(
        request_id="request",
        kv_transfer_params={"request_live_split": True},
    )

    delay_free, returned = _ascend_adapter_method("request_finished")(
        adapter, request, []
    )

    assert delay_free is True
    assert returned is None
    assert request.kv_transfer_params["ascend_live_split_source_v1"] == {
        "descriptors": [descriptor]
    }

def test_duplicate_live_source_rank_falls_back(monkeypatch) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "request_finished",
        lambda *_args: (False, {"persistent": True}),
    )
    descriptor = {"tp_rank": 0, "dp_rank": 1}
    adapter = _ascend_adapter_fake(
        _scheduler_live_sources={"request": [descriptor, dict(descriptor)]},
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                data_parallel_rank_local=0,
                data_parallel_index=1,
            )
        ),
        store_async=True,
        kv_role="kv_producer",
    )
    request = SimpleNamespace(
        request_id="request",
        kv_transfer_params={"request_live_split": True},
    )

    delay_free, params = _ascend_adapter_method("request_finished")(
        adapter, request, []
    )

    assert delay_free is True
    assert params == {"persistent": True}

def test_early_live_metadata_filters_stale_and_reuse_clears_offer(
    monkeypatch,
) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    stale = {"tp_rank": 0, "dp_rank": 0}
    fresh = {"tp_rank": 1, "dp_rank": 0}
    adapter = _ascend_adapter_fake(
        _scheduler_live_sources={},
        _unfinished_requests={"request": SimpleNamespace()},
    )
    metadata_cls = _ascend_adapter_method(
        "update_connector_worker_metadata"
    ).__globals__["LiveSourceWorkerMetadata"]

    _ascend_adapter_method("update_connector_worker_metadata")(
        adapter,
        metadata_cls({"stale": [stale], "request": [fresh]}),
        {"request"},
    )
    assert adapter._scheduler_live_sources == {"request": [fresh]}

    # The final prefiller output can mark the request finished before worker
    # metadata is ingested.  LMCache still tracks it until request_finished(),
    # so retain its descriptor even when vLLM's active set is already empty.
    late = {"tp_rank": 2, "dp_rank": 0}
    _ascend_adapter_method("update_connector_worker_metadata")(
        adapter,
        metadata_cls({"request": [late]}),
        set(),
    )
    assert adapter._scheduler_live_sources == {"request": [late]}

    request = SimpleNamespace(request_id="request")
    adapter._scheduler_live_sources["request"] = [stale]
    adapter._unfinished_requests = {}
    calls = []
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "update_state_after_alloc",
        lambda *_args: calls.append("parent"),
    )
    _ascend_adapter_method("update_state_after_alloc")(adapter, request, 0)
    assert "request" not in adapter._scheduler_live_sources
    assert calls == ["parent"]

def test_live_source_fails_closed_for_context_parallel(caplog) -> None:
    calls = []
    engine = SimpleNamespace(
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            ("store", kwargs["final"])
        ),
    )
    adapter = SimpleNamespace(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                prefill_context_parallel_size=2,
                decode_context_parallel_size=1,
            )
        ),
        _refresh_kvcaches_list=lambda: None,
        _kvcaches_for_group=lambda group: [f"cache-{group}"],
        _windowed_sparse_save_mapping=lambda request, group, base: (
            request.indexer_slot_mapping[0]
            if group
            else request.slot_mapping[0]
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=list(range(4)),
        slot_mapping=["latent"],
        indexer_slot_mapping=["index"],
        save_slot_mapping_base=0,
        save_spec=None,
        request_configs=None,
        load_spec=None,
        live_source_requested=True,
        is_last_prefill=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(adapter, [request])

    assert calls == [("store", True)]
    assert "PP/PCP/DCP must all equal 1" in caplog.text

def test_live_source_captures_rank_that_skips_persistent_store() -> None:
    captured = []
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda _req_id, _groups=(0, 1): None,
        capture_live_source_step=lambda *args: captured.append(args),
        finalize_live_source_descriptor=lambda *_args: True,
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *_args, **_kwargs: pytest.fail(
            "skip-save rank must not persist"
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: ({}, {}, 0),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[],
        live_source_token_ids=[1, 2, 3],
        live_source_slot_mapping=["live-latent"],
        live_source_indexer_slot_mapping=["live-indexer"],
        live_source_requested=True,
        load_spec=None,
        request_configs=None,
        is_last_prefill=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], source_ready_event=object()
    )

    assert captured[0][1] == [1, 2, 3]
    assert captured[0][2] == {0: ["latent"], 1: ["indexer"]}
    assert captured[0][3] == {0: "live-latent", 1: "live-indexer"}

def test_enabled_live_latent_source_is_tp0_only(monkeypatch) -> None:
    calls = []
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda req_id, groups=(0, 1): calls.append(
            (req_id, groups)
        ),
        direct_prefill_store_enabled=lambda: False,
        capture_live_source_step=lambda *_args: None,
    )
    adapter = SimpleNamespace(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _refresh_kvcaches_list=lambda: None,
        _kvcaches_for_group=lambda group: [f"cache-{group}"],
        _windowed_sparse_save_mapping=lambda request, group, _base: (
            request.indexer_slot_mapping[0]
            if group
            else request.slot_mapping[0]
        ),
        _live_latent_split_requested=True,
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=list(range(4)),
        slot_mapping=["latent"],
        indexer_slot_mapping=["index"],
        save_slot_mapping_base=0,
        save_spec=None,
        request_configs=None,
        load_spec=None,
        live_source_requested=True,
        is_last_prefill=False,
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )
    assert calls == [("request", (0, 1))]

    calls.clear()
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )
    assert calls == [("request", (1,))]

def test_final_group1_only_rank_fences_persistence() -> None:
    calls = []
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda _req_id, _groups=(0, 1): None,
        capture_live_source_step=lambda *_args: None,
        finalize_live_source_descriptor=lambda *_args: True,
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            (args[0], kwargs["final"])
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {1: ["indexer"]},
            {1: "indexer-slots"},
            0,
        ),
        _live_latent_split_requested=False,
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2, 3],
        live_source_token_ids=[1, 2, 3],
        live_source_slot_mapping=["live-latent"],
        live_source_indexer_slot_mapping=["live-indexer"],
        live_source_requested=True,
        load_spec=None,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-tp0:12345"
        },
        is_last_prefill=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], source_ready_event=object()
    )

    assert calls == [("request", True)]
    assert adapter._unfenced_live_stores == {}

def test_unqualified_remote_fill_fences_final_live_persistence(monkeypatch) -> None:
    calls = []
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda _req_id, _groups=(0, 1): None,
        capture_live_source_step=lambda *_args: None,
        finalize_live_source_descriptor=lambda *_args: True,
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            (args[0], kwargs["final"])
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _remote_store_requested=True,
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["latent"], 1: ["indexer"]},
            {0: "latent-slots", 1: "indexer-slots"},
            0,
        ),
        _live_latent_split_requested=True,
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2, 3],
        live_source_token_ids=[1, 2, 3],
        live_source_slot_mapping=["live-latent"],
        live_source_indexer_slot_mapping=["live-indexer"],
        live_source_requested=True,
        load_spec=None,
        request_configs={},
        is_last_prefill=True,
        _lmcache_remote_fill_qualified=False,
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], source_ready_event=object()
    )

    assert calls == [("request", True)]
    assert adapter._unfenced_live_stores == {}

def test_preferred_group0_store_is_fenced_before_group1_live_publish() -> None:
    calls = []
    source_ready_event = object()
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda req_id, groups=(0, 1): calls.append(
            ("begin", req_id, groups)
        ),
        capture_live_source_step=lambda *args: calls.append(("capture", args[0])),
        finalize_live_source_descriptor=lambda *_args: True,
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            (
                "store",
                args[0],
                kwargs["final"],
                kwargs["source_ready_event"],
            )
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["latent"], 1: ["indexer"]},
            {0: "latent-slots", 1: "indexer-slots"},
            0,
        ),
        _live_latent_split_requested=False,
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2, 3],
        live_source_token_ids=[1, 2, 3],
        live_source_slot_mapping=["live-latent"],
        live_source_indexer_slot_mapping=["live-indexer"],
        live_source_requested=True,
        load_spec=None,
        request_configs={
            "lmcache.mooncake_preferred_segment": "decoder-tp0:12345"
        },
        is_last_prefill=True,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter,
        [request],
        source_ready_event=source_ready_event,
        source_ready_event_source="reshape_cache_event",
    )

    assert calls == [
        ("begin", "request", (1,)),
        ("capture", "request"),
        ("store", "request", True, source_ready_event),
    ]
    assert adapter._unfenced_live_stores == {}
    assert adapter._finalized_live_source_submissions == {"request"}

def test_adopted_direct_store_still_captures_live_source() -> None:
    calls = []
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda req_id, groups=(0, 1): calls.append(
            ("begin", req_id, groups)
        ),
        direct_prefill_store_enabled=lambda: True,
        capture_live_source_step=lambda *args: calls.append(("capture", args[0])),
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            ("store", args[0], kwargs["final"])
        ),
    )
    adapter = SimpleNamespace(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _refresh_kvcaches_list=lambda: None,
        _kvcaches_for_group=lambda group: [f"cache-{group}"],
        _windowed_sparse_save_mapping=lambda request, group, base: (
            request.indexer_slot_mapping[0]
            if group
            else request.slot_mapping[0]
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=list(range(4)),
        slot_mapping=["latent"],
        indexer_slot_mapping=["index"],
        save_slot_mapping_base=0,
        save_spec=None,
        request_configs=None,
        load_spec=None,
        live_source_requested=True,
        is_last_prefill=False,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], {"request"}
    )

    assert calls == [
        ("begin", "request", (1,)),
        ("capture", "request"),
        ("store", "request", False),
    ]

def test_abort_save_step_drops_only_failed_direct_state(monkeypatch) -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    parent_calls = []
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_abort_save_step",
        lambda _self, requests: parent_calls.append(tuple(requests)),
    )
    engine_calls = []
    adapter = _ascend_adapter_fake(
        _completed_layerwise_stores={
            ("failed", 0): object(),
            ("failed", 1): object(),
            ("other", 0): object(),
        },
        lmcache_engine=SimpleNamespace(
            wait_for_direct_stores=lambda req_ids: engine_calls.append(
                ("wait", set(req_ids))
            ),
            drop_direct_store_states=lambda req_ids: engine_calls.append(
                ("drop", set(req_ids))
            ),
        ),
    )
    request = SimpleNamespace(req_id="failed")

    adapter._abort_save_step((request,))

    assert parent_calls == [(request,)]
    assert set(adapter._completed_layerwise_stores) == {("other", 0)}
    assert engine_calls == [
        ("wait", {"failed"}),
        ("drop", {"failed"}),
    ]

def test_finish_save_batch_discards_adoption_after_store_failure() -> None:
    def fail_wait():
        raise RuntimeError("store failed")

    adapter = SimpleNamespace(
        kv_role="kv_both",
        lmcache_engine=SimpleNamespace(
            wait_for_pending_sync_stores=fail_wait
        ),
        _completed_layerwise_stores={
            ("request", 0): LayerwiseStoreResult(
                request_id="request", committed_end=128
            )
        },
        _direct_store_observed_layers=set(),
    )

    with pytest.raises(RuntimeError, match="store failed"):
        _ascend_adapter_method("_finish_save_batch")(adapter, {})

    assert adapter._completed_layerwise_stores == {}

def test_finished_live_store_is_fenced_before_report_and_commit() -> None:
    request = SimpleNamespace(
        req_id="request", token_ids=[1, 2], request_configs=None
    )
    calls = []
    engine = SimpleNamespace(
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            ("finalize", args[0])
        ),
        direct_store_committed_ends=lambda _req_id: {0: 128, 1: 128},
        get_finished_stores=lambda req_ids: calls.append(("poll", set(req_ids)))
        or {"request"},
        drop_direct_store_states=lambda req_ids: calls.append(("drop", set(req_ids))),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        store_async=True,
        _unfenced_live_stores={"request": request},
        _direct_group_caches=lambda: {0: ["cache-0"], 1: ["cache-1"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["cache-0"], 1: ["cache-1"]},
            {0: "mapping-0", 1: "mapping-1"},
            0,
        ),
        _record_prefill_save_group_completed=lambda *_args: None,
        _mark_prefill_committed=lambda req: calls.append(("commit", req.req_id)),
        _release_finished_worker_requests=lambda req_ids: calls.append(
            ("release", set(req_ids))
        ),
    )

    result = _ascend_adapter_method("_finalize_worker_requests_after_store")(
        adapter, {"request"}
    )

    assert result == {"request"}
    assert calls[0] == ("finalize", "request")
    assert calls[2] == ("commit", "request")
    assert calls[3] == ("poll", {"request"})
    assert adapter._unfenced_live_stores == {}

def test_deferred_live_store_failure_does_not_block_other_requests() -> None:
    failed = SimpleNamespace(req_id="failed", token_ids=[1], request_configs=None)
    good = SimpleNamespace(req_id="good", token_ids=[1], request_configs=None)
    calls = []

    def finalize(req_id, *_args, **_kwargs):
        calls.append(("finalize", req_id))
        if req_id == "failed":
            raise RuntimeError("persistent store failed")

    engine = SimpleNamespace(
        store_direct_prefill=finalize,
        direct_store_committed_ends=lambda _req_id: {},
        drop_direct_store_states=lambda req_ids: calls.append(
            ("drop", set(req_ids))
        ),
        wait_for_pending_stores=lambda req_ids: calls.append(
            ("wait", set(req_ids))
        ),
        get_finished_stores=lambda _req_ids: {"good"},
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        store_async=True,
        _unfenced_live_stores={"failed": failed, "good": good},
        _direct_group_caches=lambda: {0: ["cache-0"], 1: ["cache-1"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["cache-0"], 1: ["cache-1"]},
            {0: "mapping-0", 1: "mapping-1"},
            0,
        ),
        _handle_save_request_error=lambda req, error: calls.append(
            ("error", req.req_id, str(error))
        ),
        _record_prefill_save_group_completed=lambda *_args: None,
        _mark_prefill_committed=lambda req: calls.append(("commit", req.req_id)),
        _release_finished_worker_requests=lambda req_ids: calls.append(
            ("release", set(req_ids))
        ),
    )

    result = _ascend_adapter_method("_finalize_worker_requests_after_store")(
        adapter, {"failed", "good"}
    )

    assert result == {"failed", "good"}
    assert ("error", "failed", "persistent store failed") in calls
    assert calls.index(("wait", {"failed"})) < calls.index(("drop", {"failed"}))
    assert ("commit", "failed") not in calls
    assert ("commit", "good") in calls
    assert ("release", {"failed", "good"}) in calls
    assert adapter._unfenced_live_stores == {}

def test_final_live_source_without_producer_event_fails_to_persistent_only(
    monkeypatch,
) -> None:
    calls = []
    engine = SimpleNamespace(
        discard_live_source_descriptor=lambda req_id: calls.append(
            ("discard", req_id)
        ),
        begin_live_source_descriptor=lambda *_args: pytest.fail(
            "unfenced live descriptor was started"
        ),
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            (
                "store",
                args[0],
                kwargs["final"],
                kwargs["source_ready_event"],
            )
        ),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {0: ["latent"], 1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["latent"], 1: ["indexer"]},
            {0: "latent-slots", 1: "indexer-slots"},
            0,
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2, 3],
        live_source_requested=True,
        load_spec=None,
        request_configs=None,
        is_last_prefill=True,
    )
    diagnostic_events = []
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "log_npu_content_diagnostic_event",
        lambda event, **fields: diagnostic_events.append((event, fields)),
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )

    assert calls == [
        ("discard", "request"),
        ("store", "request", True, None),
    ]
    assert adapter._finalized_live_source_submissions == set()
    assert diagnostic_events == [
        (
            "group1_source_missing_producer_event",
            {
                "req_id": "request",
                "event_source": "missing",
                "action": "persistent_only",
                "fallback_event_created": False,
            },
        )
    ]

def test_save_batch_does_not_rebuild_completed_live_descriptor() -> None:
    request = SimpleNamespace(req_id="request")
    adapter = _ascend_adapter_fake(
        lmcache_engine=SimpleNamespace(),
        _unfenced_live_stores={},
        _live_source_ready_fences={},
        _finalized_live_source_submissions={"request"},
        _direct_group_caches=lambda: pytest.fail(
            "completed live request was submitted twice"
        ),
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], source_ready_event=object()
    )

def test_source_ready_event_rejects_ambiguous_indexer_sibling() -> None:
    metadata = {
        "model.layers.0.self_attn.attn": SimpleNamespace(
            reshape_cache_event=object()
        ),
        "model.layers.0.self_attn.mla": SimpleNamespace(
            reshape_cache_event=object()
        ),
    }

    source_event = _ascend_adapter_method("_source_ready_event")
    assert (
        source_event(
            "model.layers.0.self_attn.indexer.k_cache",
            metadata,
        )
        is None
    )

def test_source_ready_event_resolves_unbundled_indexer_sibling() -> None:
    expected = object()
    metadata = {
        "model.layers.0.self_attn.attn": SimpleNamespace(
            reshape_cache_event=expected
        ),
        "model.layers.1.self_attn.attn": SimpleNamespace(
            reshape_cache_event=object()
        ),
    }

    source_event = _ascend_adapter_method("_source_ready_event")
    assert (
        source_event(
            "model.layers.0.self_attn.indexer.k_cache",
            metadata,
        )
        is expected
    )

def test_source_ready_event_uses_matching_layer_metadata() -> None:
    expected = object()
    metadata = {
        "layer.0": SimpleNamespace(reshape_cache_event=object()),
        "layer.1": SimpleNamespace(reshape_cache_event=expected),
    }

    source_event = _ascend_adapter_method("_source_ready_event")
    assert source_event("layer.1", metadata) is expected
    assert source_event("missing", metadata) is None

def test_save_layer_retains_callback_fence_for_remote_fill_finish() -> None:
    event = object()
    request = SimpleNamespace(
        req_id="request",
        _lmcache_remote_fill_qualified=True,
    )
    adapter = _ascend_adapter_fake(
        config=SimpleNamespace(dsa_two_groups=False),
        _remote_store_requested=True,
        _latent_layer_names=("latent",),
        _indexer_layer_names=(),
        _direct_prefill_requests=lambda: [request],
        _preflight_direct_store=lambda _requests: True,
        _submit_direct_prefill_requests=lambda *_args, **_kwargs: None,
    )

    _ascend_adapter_method("save_kv_layer")(
        adapter,
        "latent",
        object(),
        SimpleNamespace(reshape_cache_event=event),
    )

    assert adapter._direct_store_observed_layers == {"latent"}
    assert adapter._latest_live_source_ready_event is event
    assert adapter._latest_direct_source_ready_events == {"latent": event}

def test_save_layer_preserves_distinct_group_producer_fences() -> None:
    latent_event = object()
    index_event = object()
    submitted = []
    request = SimpleNamespace(req_id="request")
    adapter = _ascend_adapter_fake(
        config=SimpleNamespace(dsa_two_groups=True),
        _latent_layer_names=("latent",),
        _indexer_layer_names=("index",),
        _direct_prefill_requests=lambda: [request],
        _preflight_direct_store=lambda _requests: True,
        _submit_direct_prefill_requests=lambda requests, **kwargs: submitted.append(
            (requests, kwargs)
        ),
        _latest_live_source_ready_event=None,
        _latest_live_source_ready_event_source="missing",
    )

    _ascend_adapter_method("save_kv_layer")(
        adapter,
        "latent",
        object(),
        SimpleNamespace(reshape_cache_event=latent_event),
    )
    _ascend_adapter_method("save_kv_layer")(
        adapter,
        "index",
        object(),
        SimpleNamespace(reshape_cache_event=index_event),
    )

    assert submitted[0][1]["source_ready_events"] == (
        latent_event,
        index_event,
    )

def test_save_layer_carries_final_indexer_producer_event() -> None:
    event = object()
    submitted = []
    request = SimpleNamespace(req_id="request")
    adapter = _ascend_adapter_fake(
        config=SimpleNamespace(dsa_two_groups=True),
        _latent_layer_names=("model.layers.0.self_attn.attn",),
        _indexer_layer_names=("model.layers.0.self_attn.indexer.k_cache",),
        _direct_prefill_requests=lambda: [request],
        _preflight_direct_store=lambda _requests: True,
        _submit_direct_prefill_requests=lambda requests, **kwargs: submitted.append(
            (requests, kwargs)
        ),
        _latest_live_source_ready_event=None,
        _latest_live_source_ready_event_source="missing",
    )
    metadata = {
        "model.layers.0.self_attn.attn": SimpleNamespace(
            reshape_cache_event=event
        )
    }

    _ascend_adapter_method("save_kv_layer")(
        adapter,
        "model.layers.0.self_attn.attn",
        object(),
        metadata,
    )
    assert submitted == []
    _ascend_adapter_method("save_kv_layer")(
        adapter,
        "model.layers.0.self_attn.indexer.k_cache",
        object(),
        metadata,
    )

    assert submitted[0][0] == [request]
    assert submitted[0][1]["source_ready_event"] is event
    assert submitted[0][1]["source_ready_event_source"] == (
        "attn_metadata.reshape_cache_event"
    )
    assert submitted[0][1]["source_ready_events"] == (event,)
    assert adapter._latest_live_source_ready_event is None

def test_singleton_readiness_does_not_claim_complete_remote_fill_fence() -> None:
    event = object()
    state = SimpleNamespace(
        source_ready_event=None,
        source_ready_event_source="missing",
        source_ready_token_end=0,
        source_ready_events=(),
        source_ready_events_token_end=0,
    )

    AscendLMCacheEngine._remember_direct_source_readiness(
        state,
        1024,
        event,
        "attn_metadata.reshape_cache_event",
    )

    assert state.source_ready_event is event
    assert state.source_ready_token_end == 1024
    assert AscendLMCacheEngine._direct_source_ready_events(state, 1024) == ()

def test_direct_store_retains_causal_join_through_exact_token_frontier() -> None:
    event = object()
    state = SimpleNamespace(
        source_ready_event=None,
        source_ready_event_source="missing",
        source_ready_token_end=0,
        source_ready_events=(),
        source_ready_events_token_end=0,
    )

    AscendLMCacheEngine._remember_direct_source_readiness(
        state,
        1024,
        event,
        "forward_context.sfa_reshape_cache_event",
        (event, event),
    )

    assert state.source_ready_events == (event,)
    assert state.source_ready_events_token_end == 1024
    assert AscendLMCacheEngine._direct_source_ready_events(state, 1024) == (
        event,
    )
    assert AscendLMCacheEngine._direct_source_ready_events(state, 1025) == ()

def test_group1_direct_store_rejects_current_stream_event_fallback() -> None:
    state = SimpleNamespace(
        source_ready_event=None,
        source_ready_event_source="missing",
        source_ready_token_end=0,
    )

    with pytest.raises(RuntimeError, match="no attention producer event"):
        AscendLMCacheEngine._direct_source_ready_event(
            state,
            128,
            require_producer_event=True,
        )

def test_discard_live_source_keeps_persistent_store_state() -> None:
    direct_state = object()
    engine = SimpleNamespace(
        _live_source_builders={"request": object()},
        _completed_live_sources={"request": object()},
        _pending_live_source_diagnostics={"request": object()},
        _direct_store_states={"request": direct_state},
    )

    AscendLMCacheEngine.discard_live_source_descriptor(engine, "request")

    assert engine._live_source_builders == {}
    assert engine._completed_live_sources == {}
    assert engine._pending_live_source_diagnostics == {}
    assert engine._direct_store_states == {"request": direct_state}

def test_descriptor_drain_rejects_unfenced_diagnostic_source() -> None:
    descriptor = {"tp_rank": 0, "dp_rank": 0}
    engine = SimpleNamespace(
        _pending_live_source_diagnostics={"request": {}},
        _completed_live_sources={"request": descriptor},
    )

    with pytest.raises(RuntimeError, match="before their post-fence"):
        AscendLMCacheEngine.drain_live_source_descriptors(engine)

    engine._pending_live_source_diagnostics.clear()
    assert AscendLMCacheEngine.drain_live_source_descriptors(engine) == {
        "request": descriptor
    }

def test_post_fence_source_fingerprint_is_attached_to_wire_descriptor(
    monkeypatch,
) -> None:
    calls = []
    fingerprint = {"content_hash": "post-fence"}

    def fake_fingerprint(**kwargs):
        calls.append(kwargs)
        return fingerprint

    monkeypatch.setattr(
        "lmcache_ascend.v1.cache_engine.fingerprint_compact_group1",
        fake_fingerprint,
    )
    descriptor = {"tp_rank": 0, "dp_rank": 1}
    pending = {
        "request": {
            "owners": [object()],
            "layers": [object()],
            "runs": [object()],
            "token_count": 17,
            "chunk_size": 4,
            "tp_rank": 0,
            "dp_rank": 1,
        }
    }
    engine = SimpleNamespace(
        _pending_live_source_diagnostics=pending,
        _completed_live_sources={"request": descriptor},
    )

    AscendLMCacheEngine.finalize_live_source_readiness(engine, ["request"])

    assert calls[0]["event"] == "group1_source_post_fence_fingerprint"
    assert calls[0]["token_count"] == 17
    assert calls[0]["dp_rank"] == 1
    assert descriptor["content_diagnostics"] is fingerprint
    assert pending == {}

def test_source_readiness_query_precedes_descriptor_finalize(monkeypatch) -> None:
    calls = []

    class FakeEvent:
        def query(self):
            calls.append("query")
            return False

    event = FakeEvent()
    engine = SimpleNamespace(
        begin_live_source_descriptor=lambda *_args: calls.append("begin"),
        capture_live_source_step=lambda *_args: calls.append("capture"),
        finalize_live_source_descriptor=lambda *_args: (
            calls.append("finalize") or True
        ),
        direct_prefill_store_enabled=lambda: False,
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
        _direct_group_caches=lambda: {1: ["indexer"]},
        _direct_request_inputs=lambda *_args: (
            {},
            {1: "live-indexer-slots"},
            0,
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=[],
        live_source_token_ids=[1, 2, 3],
        live_source_slot_mapping=None,
        live_source_indexer_slot_mapping=["live-indexer-slots"],
        live_source_requested=True,
        load_spec=None,
        request_configs=None,
        is_last_prefill=True,
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "npu_content_diagnostics_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter."
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter,
        [request],
        source_ready_event=event,
        source_ready_event_source="test",
    )

    assert calls == ["begin", "capture", "query", "finalize"]
    assert adapter._live_source_ready_fences["request"].ready_at_finalize is False

def test_live_source_event_is_fenced_before_descriptor_drain(monkeypatch) -> None:
    calls = []

    class FakeEvent:
        ready = False

        def query(self):
            calls.append(("query", self.ready))
            return self.ready

        def synchronize(self):
            calls.append("event-synchronize")
            self.ready = True

    event = FakeEvent()
    descriptor = {"tp_rank": 0, "dp_rank": 0}

    class FakeEngine:
        def finalize_live_source_readiness(self, req_ids):
            calls.append(("post-fence-fingerprint", list(req_ids), event.ready))

        def drain_live_source_descriptors(self):
            calls.append(("drain", event.ready))
            return {"request": descriptor}

    module = __import__(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter",
        fromlist=["_LiveSourceReadyFence"],
    )
    monkeypatch.setattr(module, "npu_content_diagnostics_enabled", lambda: True)
    diagnostic_events = []
    monkeypatch.setattr(
        module,
        "log_npu_content_diagnostic_event",
        lambda name, **fields: diagnostic_events.append((name, fields)),
    )
    fence = module._LiveSourceReadyFence(
        event=event,
        event_source="attn_metadata.reshape_cache_event",
        ready_at_finalize=False,
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=FakeEngine(),
        _live_source_ready_fences={"request": fence},
    )

    metadata = _ascend_adapter_method("build_connector_worker_meta")(adapter)

    assert calls == [
        ("query", False),
        "event-synchronize",
        ("post-fence-fingerprint", ["request"], True),
        ("drain", True),
    ]
    assert metadata.descriptors == {"request": [descriptor]}
    assert adapter._live_source_ready_fences == {}
    assert diagnostic_events[0][0] == "group1_source_ready_fence"
    assert diagnostic_events[0][1]["ready_at_finalize"] is False
    assert diagnostic_events[0][1]["query_precedes_device_readback"] is True
    assert diagnostic_events[0][1]["ready_after_fence"] is True

def test_live_final_metadata_is_built_before_persistent_final_fence() -> None:
    calls = []
    request = SimpleNamespace(
        req_id="request",
        token_ids=[1, 2],
        request_configs=None,
    )
    engine = SimpleNamespace(
        drain_live_source_descriptors=lambda: calls.append("metadata") or {},
        store_direct_prefill=lambda *args, **kwargs: calls.append(
            f"store-final-{kwargs['final']}"
        ),
        direct_store_committed_ends=lambda _req_id: {},
        get_finished_stores=lambda _req_ids: set(),
    )
    adapter = _ascend_adapter_fake(
        lmcache_engine=engine,
        store_async=True,
        _unfenced_live_stores={"request": request},
        _direct_group_caches=lambda: {0: ["cache-0"], 1: ["cache-1"]},
        _direct_request_inputs=lambda *_args: (
            {0: ["cache-0"], 1: ["cache-1"]},
            {0: "mapping-0", 1: "mapping-1"},
            0,
        ),
        _record_prefill_save_group_completed=lambda *_args: None,
        _mark_prefill_committed=lambda *_args: None,
    )

    _ascend_adapter_method("build_connector_worker_meta")(adapter)
    _ascend_adapter_method("_finalize_worker_requests_after_store")(
        adapter, {"request"}
    )

    assert calls[:2] == ["metadata", "store-final-True"]

def test_final_live_store_is_not_resubmitted_by_finish_batch() -> None:
    request = SimpleNamespace(req_id="request")
    adapter = _ascend_adapter_fake(
        lmcache_engine=SimpleNamespace(),
        _direct_group_caches=lambda: (_ for _ in ()).throw(
            AssertionError("resubmitted final live request")
        ),
        _unfenced_live_stores={"request": request},
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request], source_ready_event=object()
    )

def test_finish_save_batch_defers_only_finalized_live_store(live: bool) -> None:
    request = SimpleNamespace(req_id="request", is_last_prefill=True)
    waited = []
    engine = SimpleNamespace(
        wait_for_pending_sync_stores=lambda: None,
        wait_for_direct_stores=lambda req_ids: waited.append(set(req_ids)),
        direct_store_committed_ends=lambda _req_id: {0: 128, 1: 128},
    )
    marked = []
    adapter = _ascend_adapter_fake(
        kv_role="kv_both",
        lmcache_engine=engine,
        _completed_layerwise_stores={},
        _direct_prefill_requests=lambda: [request],
        _submit_direct_prefill_requests=lambda *_args: None,
        _unfenced_live_stores={"request": request} if live else {},
        _record_prefill_save_group_completed=lambda *_args: None,
        _mark_prefill_committed=lambda req: marked.append(req.req_id),
    )

    _ascend_adapter_method("_finish_save_batch")(adapter, {})

    assert waited == ([] if live else [{"request"}])
    assert marked == ([] if live else ["request"])

def test_finish_save_batch_preserves_final_attention_producer_event() -> None:
    request = SimpleNamespace(req_id="request", is_last_prefill=True)
    event = object()
    submitted = []
    waited = []
    marked = []
    adapter = _ascend_adapter_fake(
        kv_role="kv_both",
        lmcache_engine=SimpleNamespace(
            wait_for_pending_sync_stores=lambda: None,
            wait_for_direct_stores=lambda req_ids: waited.append(
                set(req_ids)
            ),
            direct_store_committed_ends=lambda _req_id: {},
        ),
        _completed_layerwise_stores={},
        _direct_prefill_requests=lambda: [request],
        _submit_direct_prefill_requests=(
            lambda requests, adopted, **kwargs: submitted.append(
                (requests, adopted, kwargs)
            )
        ),
        _latest_live_source_ready_event=event,
        _latest_live_source_ready_event_source=(
            "attn_metadata.reshape_cache_event"
        ),
        _mark_prefill_committed=lambda req: marked.append(req.req_id),
    )

    _ascend_adapter_method("_finish_save_batch")(adapter, {})

    assert submitted == [
        (
            [request],
            set(),
            {
                "source_ready_event": event,
                "source_ready_event_source": (
                    "attn_metadata.reshape_cache_event"
                ),
            },
        )
    ]
    assert adapter._latest_live_source_ready_event is None
    assert adapter._latest_live_source_ready_event_source == "missing"
    assert waited == [{"request"}]
    assert marked == ["request"]

def test_finish_save_batch_submits_nonfinal_direct_window() -> None:
    request = SimpleNamespace(req_id="request", is_last_prefill=False)
    result = LayerwiseStoreResult(
        request_id="request", committed_end=128
    )
    calls = []
    adapter = SimpleNamespace(
        kv_role="kv_both",
        lmcache_engine=SimpleNamespace(
            wait_for_pending_sync_stores=lambda: calls.append("wait"),
            adopt_completed_layerwise_store=lambda adopted: calls.append(adopted),
        ),
        _completed_layerwise_stores={("request", 0): result},
        _direct_store_observed_layers=set(),
        _direct_prefill_requests=lambda: [request],
        _submit_direct_prefill_requests=(
            lambda requests, adopted, **kwargs: calls.append(
                (requests, adopted, kwargs)
            )
        ),
    )

    _ascend_adapter_method("_finish_save_batch")(adapter, {})

    assert calls == [
        "wait",
        result,
        (
            [request],
            {"request"},
            {
                "source_ready_event": None,
                "source_ready_event_source": "missing",
            },
        ),
    ]
    assert adapter._completed_layerwise_stores == {}

def test_direct_prefill_uses_window_relative_save_mappings() -> None:
    calls = []
    mapping_calls = []
    engine = SimpleNamespace(
        direct_prefill_store_enabled=lambda: True,
        store_direct_prefill=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    adapter = SimpleNamespace(
        lmcache_engine=engine,
        config=SimpleNamespace(dsa_two_groups=True),
        _refresh_kvcaches_list=lambda: None,
        _kvcaches_for_group=lambda group: [f"cache-{group}"],
        _windowed_sparse_save_mapping=lambda request, group, base: (
            mapping_calls.append(group)
            or (
                request.save_indexer_slot_mapping[0]
                if group
                else request.save_slot_mapping[0]
            )
        ),
    )
    request = SimpleNamespace(
        req_id="request",
        token_ids=list(range(512)),
        slot_mapping=["full-latent"],
        indexer_slot_mapping=["full-indexer"],
        save_slot_mapping=["window-latent"],
        save_indexer_slot_mapping=["window-indexer"],
        save_slot_mapping_base=256,
        save_spec=None,
        request_configs=None,
        load_spec=None,
        is_last_prefill=False,
    )

    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )

    assert calls[0][0][3] == {
        0: "window-latent",
        1: "window-indexer",
    }
    assert calls[0][1]["slot_mapping_base"] == 256
    assert calls[0][1]["verified_prefix_end"] == 0
    assert calls[0][1]["accepted_store_end"] == 512

    request.load_spec = SimpleNamespace(lmcache_cached_tokens=256)
    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )
    assert calls[-1][1]["verified_prefix_end"] == 256

    adapter._windowed_sparse_save_mapping = (
        lambda request, group, base: None
        if group
        else request.save_slot_mapping[0]
    )
    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )
    assert calls[-1][0][3] == {0: "full-latent", 1: "full-indexer"}
    assert calls[-1][1]["slot_mapping_base"] == 0

    mapping_calls.clear()
    adapter._windowed_sparse_save_mapping = lambda request, group, base: (
        mapping_calls.append(group) or request.save_indexer_slot_mapping[0]
    )
    request.save_spec = SimpleNamespace(
        can_save_latent=False, can_save_indexer=True
    )
    request.save_slot_mapping = []
    _ascend_adapter_method("_submit_direct_prefill_requests")(
        adapter, [request]
    )
    assert mapping_calls == [1]
    assert calls[-1][0][2] == {1: ["cache-1"]}

def _ascend_adapter_fake(**attrs):
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )

    fake = object.__new__(LMCacheAscendConnectorV1Impl)
    fake._finished_req_ids_waiting_for_save = set()
    fake._late_finished_sending = set()
    fake._direct_store_observed_layers = set()
    fake._direct_store_step_supported = None
    fake._unfenced_live_stores = {}
    fake._latest_live_source_ready_event = None
    fake._latest_live_source_ready_event_source = "missing"
    fake._live_source_ready_fences = {}
    fake._finalized_live_source_submissions = set()
    for name, value in attrs.items():
        setattr(fake, name, value)
    return fake

def test_page_store_pointer_cache_does_not_rebuild_layer_views() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=2,
        num_layers=1,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=8,
        full_tokens=8,
    )
    assert pages is not None
    pointer_sources = []
    engine = SimpleNamespace(
        gpu_connector=SimpleNamespace(
            append_sparse_chunk_ptr_cache_for_layer=(
                lambda _layer, sources, *_cache: pointer_sources.extend(sources)
            )
        )
    )
    cached_tensors: list[list] = []

    AscendLMCacheEngine._append_layer_store_tensors(
        engine,
        0,
        [pages],
        cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
    )

    assert pointer_sources == pages
    assert cached_tensors == []
    for page in pages:
        page.ref_count_down()

def test_layer_cache_publication_rejects_missing_tensor() -> None:
    with pytest.raises(ValueError, match="Layerwise cache source has no tensor"):
        AscendLMCacheEngine._layer_memory_tensor(
            SimpleNamespace(tensor=None), 0
        )

def test_retrieve_fallback_selects_page_layer_view() -> None:
    tensor = torch.tensor([2])
    page = MagicMock(spec=LayerPageMemoryObj)
    page.layer_tensor.return_value = tensor
    cached_memory_objs: list[list] = []
    cached_tensors: list[list] = []
    engine = SimpleNamespace(gpu_connector=SimpleNamespace(), num_layers=2)

    AscendLMCacheEngine._append_retrieve_layer_cache(
        engine,
        1,
        [page],
        cached_memory_objs,
        cached_tensors,
        None,
        None,
    )

    page.layer_tensor.assert_called_once_with(1)
    assert cached_tensors[1] == [tensor]

def test_group_store_page_publication_reuses_pointer_table_without_views() -> None:
    page = MagicMock(spec=LayerPageMemoryObj)
    memory_objs = [[page], [page]]
    cached_tensors = [[], []]
    cached_chunk_dev_ptrs: list[list[int]] = []
    cached_chunk_ptrs_npu: list[torch.Tensor | None] = []
    pointer_table = torch.tensor([[101], [202]], dtype=torch.long)

    AscendLMCacheEngine._append_group_store_tensors(
        SimpleNamespace(),
        memory_objs,
        cached_tensors,
        cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu,
        [[101], [202]],
        pointer_table,
    )

    assert cached_tensors == [[], []]
    page.layer_tensor.assert_not_called()
    assert cached_chunk_dev_ptrs == [[101], [202]]
