# SPDX-License-Identifier: Apache-2.0
"""Execute native owner contracts with host-only tensor/stream collaborators."""

from __future__ import annotations

# Standard
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import importlib.util
import logging

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]


def functions(path: str, names: list[str], **namespace) -> dict:
    """Load exact production ASTs; fake only external dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = []
    for name in names:
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        node.decorator_list = []
        nodes.append(node)
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[prefix, *nodes], type_ignores=[])
    )
    exec(compile(module, path, "exec"), namespace)
    return namespace


def test_static_native_gate() -> None:
    spec = importlib.util.spec_from_file_location(
        "p3_native_gate", ROOT / "tools/check_npu_native.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    result = gate.check()
    assert result["passed"], result["errors"]
    assert result["merged_classes"] == 6  # P4 removes two SGLang + one Blend owner.


class Tensor:
    """Only shape, dtype and device identity used by these host contracts."""

    def __init__(self, shape=(), dtype="bf16", device="npu"):
        self.shape, self.dtype, self.device = shape, dtype, NS(type=device)


@pytest.mark.parametrize(
    "shape,expected", [((2, 16, 2, 4, 8), 32), ((2, 16, 4, 8), 32), ((2, 16, 576), 576)]
)
def test_group_hidden_dimension(shape, expected) -> None:
    ns = functions("lmcache/v1/kv_layer_groups.py", ["hidden_dim_size"])
    assert ns["hidden_dim_size"](NS(shape=shape)) == expected


def test_tuple_grouping_keeps_mtp_cardinality_and_order() -> None:
    ns = functions(
        "lmcache/v1/kv_layer_groups.py",
        [
            "_get_tuple_storage_shape",
            "_get_kv_cache_group_key_and_info",
            "build_kv_layer_groups",
        ],
        torch=NS(Tensor=Tensor, Size=tuple),
        defaultdict=defaultdict,
        KVLayerGroupInfo=lambda **kw: NS(**kw),
        logger=logging.getLogger(__name__),
    )
    caches = {
        "latent.0": (Tensor((2, 16, 1, 512)), Tensor((2, 16, 1, 64))),
        "index.0": (Tensor((2, 16, 1, 128)),),
        "latent.mtp": (Tensor((2, 16, 1, 512)), Tensor((2, 16, 1, 64))),
    }
    manager = NS(kv_layer_groups=[])
    ns["build_kv_layer_groups"](manager, caches)
    assert [g.shape for g in manager.kv_layer_groups] == [(2, 16, 576), (2, 16, 128)]
    assert [len(g.layer_names) for g in manager.kv_layer_groups] == [2, 1]
    previous = list(manager.kv_layer_groups)
    ns["build_kv_layer_groups"](manager, caches)
    assert manager.kv_layer_groups == previous
    with pytest.raises(ValueError, match="mixed dtypes"):
        ns["_get_kv_cache_group_key_and_info"](
            (Tensor((2, 16, 1, 512)), Tensor((2, 16, 1, 64), "fp32"))
        )


def test_native_payload_event_uses_producer_stream_without_cuda_alias() -> None:
    calls = []
    event = NS(record=lambda stream: calls.append(stream))
    fake = NS(
        Tensor=Tensor, npu=NS(Event=lambda: event, current_stream=lambda: "producer")
    )
    ns = functions(
        "lmcache/integration/vllm/vllm_v1_adapter.py",
        ["_dsa_device_tensor_types", "_dsa_record_payload_event_if_needed"],
        torch=fake,
    )
    record = ns["_dsa_record_payload_event_if_needed"]
    assert record(Tensor(device="cpu")) is None
    assert record(Tensor(), [Tensor(device="cpu")]) is event
    assert calls == ["producer"]
    with pytest.raises(RuntimeError, match="unsupported"):
        record(Tensor(device="cuda"))


def test_checkpoint_output_reaches_preserved_common_method() -> None:
    ns = functions(
        "lmcache/integration/vllm/vllm_v1_adapter.py", ["update_connector_output"]
    )
    calls = []
    ns["update_connector_output"](
        NS(_common_update_connector_output=calls.append), "completion"
    )
    assert calls == ["completion"]


def test_hash_key_identity_and_extra_keys_are_unchanged() -> None:
    ns = functions(
        "lmcache/v1/token_database.py", ["_hash_tokens"], torch=NS(Tensor=Tensor)
    )
    engine = NS(hash_func=lambda value: value)
    assert ns["_hash_tokens"](engine, [1, 2], 7) == (7, (1, 2))
    assert ns["_hash_tokens"](engine, [1, 2], 7, ("scope",)) == (7, (1, 2), ("scope",))
    with pytest.raises(ValueError):
        ns["_hash_tokens"](engine, (1, 2))


def test_rpc_socket_shortens_engine_id_exactly_once() -> None:
    ns = functions(
        "lmcache/v1/rpc_utils.py",
        ["get_zmq_rpc_path_lmcache"],
        logger=logging.getLogger(__name__),
    )
    result = ns["get_zmq_rpc_path_lmcache"](
        "long-engine-id", rank=3, base_url="/tmp/p3"
    )
    short = hashlib.md5(b"long-engine-id").hexdigest()[:8]
    assert short in result
    assert hashlib.md5(short.encode()).hexdigest()[:8] not in result


@pytest.mark.parametrize(
    "device,expected", [("cpu", "cpu"), ("npu", "npu:3"), ("npu:0", "npu:3")]
)
def test_transfer_device_resolution(device, expected) -> None:
    ns = functions(
        "lmcache/v1/transfer_channel/transfer_utils.py", ["get_correct_device"]
    )
    assert ns["get_correct_device"](device, 3) == expected
    with pytest.raises(ValueError):
        ns["get_correct_device"]("cuda", 0)


def test_native_ipc_reconstruction_uses_explicit_npu_device() -> None:
    calls = []
    tensor = NS(set_=lambda *args: calls.append(args))
    fake = NS(
        UntypedStorage=NS(_new_shared_npu=lambda *args: ("storage", args)),
        empty=lambda *args, **kwargs: (calls.append(kwargs), tensor)[1],
    )
    ns = functions(
        "lmcache/v1/multiprocess/custom_types.py",
        ["to_tensor"],
        torch=fake,
        NPUIPCWrapper=NS(_get_device_index_from_uuid=lambda _: 3),
    )
    wrapper = NS(
        device_uuid="uuid",
        handle=(0, "handle"),
        dtype="bf16",
        storage_offset=0,
        shape=(2, 3),
        stride=(3, 1),
    )
    assert ns["to_tensor"](wrapper) is tensor
    assert calls[0]["device"] == "npu:3"


def test_pinned_allocator_keeps_registered_host_ownership() -> None:
    ops = NS(
        alloc_shm_pinned_ptr=object(),
        free_shm_pinned_ptr=object(),
        alloc_pinned_numa_ptr=object(),
        free_pinned_numa_ptr=object(),
        alloc_pinned_ptr=object(),
        free_pinned_ptr=object(),
    )
    ns = functions(
        "lmcache/v1/memory_management.py",
        ["_resolve_pinned_alloc_free"],
        lmc_ops=ops,
        torch=NS(npu=NS(is_available=lambda: True, current_device=lambda: 3)),
    )
    resolve = ns["_resolve_pinned_alloc_free"]
    alloc, free = resolve(shm_name="p3", size=4096, shm_interleave_nodes=(0, 1))
    assert alloc == (ops.alloc_shm_pinned_ptr, "p3", [0, 1])
    assert free == (ops.free_shm_pinned_ptr, 4096, "p3")
    assert resolve(numa_mapping=NS(gpu_to_numa_mapping={3: 1}), size=8192) == (
        (ops.alloc_pinned_numa_ptr, 1),
        (ops.free_pinned_numa_ptr, 8192),
    )


def test_cpp_layout_enum_is_defined_without_python_injection() -> None:
    source = (ROOT / "csrc/pybind.cpp").read_text()
    for value, name in enumerate(
        [
            "NB_NL_TWO_BS_NH_HS",
            "NL_X_TWO_NB_BS_NH_HS",
            "NL_X_NB_TWO_BS_NH_HS",
            "NL_X_NB_BS_HS",
            "TWO_X_NL_X_NBBS_NH_HS",
            "NL_X_NBBS_ONE_HS",
            "NL_X_TWO_NB_NH_BS_HS",
            "NL_X_NB_TWO_NH_BS_HS",
        ]
    ):
        assert f"{name} = {value}" in source
        assert f'.value("{name}", GPUKVFormat::{name})' in source


def test_mooncake_connector_borrows_native_transfer_engine() -> None:
    requests = []
    engine = NS(get_rpc_port=lambda: 12345, get_engine=lambda: "native-engine")
    global_te = NS(get_transfer_engine=lambda *_args, **_kwargs: engine)

    def import_module(name):
        requests.append(name)
        if name == "vllm.distributed.kv_transfer.ascend.utils.mooncake_transfer_engine":
            return NS(global_te=global_te)
        if name == "vllm.utils.network_utils":
            return NS(get_ip=lambda: "10.0.0.1")
        raise ModuleNotFoundError(name)

    ns = functions(
        "lmcache/v1/storage_backend/connector/mooncakestore_connector.py",
        ["_shared_vllm_mooncake_transport"],
        import_module=import_module,
    )
    assert ns["_shared_vllm_mooncake_transport"]() == (
        global_te,
        "10.0.0.1:12345",
        "native-engine",
        engine,
    )
    assert requests[0] == (
        "vllm.distributed.kv_transfer.ascend.utils.mooncake_transfer_engine"
    )


def test_registered_buffer_copies_keep_nonblocking_semantics() -> None:
    ns = functions(
        "lmcache/v1/device_connector/memory_ops.py",
        ["lmcache_memcpy_async_h2d", "lmcache_memcpy_async_d2h"],
    )
    copies = []
    host = NS(
        numel=lambda: 4, copy_=lambda source, **kw: copies.append(("host", source, kw))
    )
    device = NS(
        numel=lambda: 4,
        copy_=lambda source, **kw: copies.append(("device", source, kw)),
    )
    memory = NS(tensor=host)
    ns["lmcache_memcpy_async_h2d"](memory, device)
    ns["lmcache_memcpy_async_d2h"](device, memory)
    assert copies == [
        ("device", host, {"non_blocking": True}),
        ("host", device, {"non_blocking": True}),
    ]
    with pytest.raises(AssertionError):
        ns["lmcache_memcpy_async_h2d"](NS(tensor=None), device)
