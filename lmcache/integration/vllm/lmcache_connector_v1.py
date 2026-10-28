# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
import torch
from lmcache.v1.device_connector.sparse import PreparedSparseSource
from typing import Any
from copy import deepcopy
import os

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


if TYPE_CHECKING:
    # Third Party
    from vllm.v1.kv_cache_interface import KVCacheConfig


class LMCacheConnectorV1Dynamic(KVConnectorBase_V1, SupportsHMA):

    @classmethod
    def get_dsa_compact_startup_policy(
        cls, vllm_config: VllmConfig
    ) -> Optional[tuple[int, int]]:
        # This module imports lmcache before LMCache's config singleton, so
        # the same extended config class is used by probing and construction.
        from lmcache.integration.vllm.utils import lmcache_get_or_create_config
        from lmcache.v1.config_base import validate_and_set_config_value

        config = deepcopy(lmcache_get_or_create_config())
        transfer = vllm_config.kv_transfer_config
        for key, value in (
            getattr(transfer, "kv_connector_extra_config", None) or {}
        ).items():
            if key.startswith("lmcache."):
                validate_and_set_config_value(config, key[8:], value)
        return cls._compact_policy_from_config(vllm_config, config)

    @staticmethod
    def _compact_policy_from_config(
        vllm_config: VllmConfig, config: Any
    ) -> Optional[tuple[int, int]]:
        transfer = vllm_config.kv_transfer_config
        extra = getattr(config, "extra_config", None)
        shared_cpu = (
            extra.get(
                "enable_shared_cpu_cache", config.enable_shared_cpu_cache
            )
            if isinstance(extra, dict)
            else config.enable_shared_cpu_cache
        )
        if (
            transfer is None
            or transfer.kv_role not in ("kv_consumer", "kv_both")
            or getattr(transfer, "kv_load_failure_policy", None) != "fail"
            or getattr(config, "pd_role", None) != "receiver"
            or getattr(config, "dsa_group1_load_mode", None)
            not in ("persistent_direct_hbm", "p2p_preferred")
            or not getattr(config, "enable_dsa_cold_compact_load", False)
            or vllm_config.cache_config.enable_prefix_caching
            or not config.enable_sparse_attention
            or not config.dsa_two_groups
            or not shared_cpu
            or not config.use_layerwise
        ):
            return None
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        topk = int(getattr(hf_config, "index_topk", 0) or 0)
        if topk <= 0:
            return None
        scratch = (
            1 + max(int(getattr(vllm_config, "num_speculative_tokens", 0)), 0)
        ) * topk
        try:
            threshold = max(
                int(os.environ.get("LMCACHE_DSA_KV_POLICY_THRESHOLD", "0")), 0
            )
        except (TypeError, ValueError):
            threshold = 0
        return scratch, threshold

    def get_dsa_compact_runtime_policy(
        self, vllm_config: VllmConfig
    ) -> Optional[tuple[int, int]]:
        engine = self._lmcache_engine
        policy = self._compact_policy_from_config(vllm_config, engine.config)
        if policy != (
            getattr(engine, "_dsa_scratch_capacity", None),
            getattr(engine, "_dsa_kv_policy_threshold", None),
        ):
            return None
        return policy

    def _common_init(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
        # Lazy import keeps the serving framework optional until construction.
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        if kv_cache_config is not None:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1Impl(
            vllm_config,
            role,
            self,
            **(
                {"kv_cache_config": kv_cache_config}
                if kv_cache_config is not None
                else {}
            ),
        )

    @property
    def supports_dsa_compact_external_load(self) -> bool:
        return self._lmcache_engine.supports_dsa_cold_compact_load()

    @property
    def supports_preemption_checkpoint(self) -> bool:
        """Whether this connector requests pre-overwrite decoder snapshots."""
        return bool(
            getattr(self._lmcache_engine.config, "decode_preemption_checkpoint", False)
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """Drain source owners before the runner reuses preempted blocks."""
        handle = getattr(self._lmcache_engine, "handle_preemptions", None)
        if callable(handle):
            handle(preempted_req_ids)

    def prepare_preemption_checkpoint(self, snapshot: tuple) -> None:
        """Forward a scheduler victim snapshot to the actual implementation."""
        self._lmcache_engine.prepare_preemption_checkpoint(snapshot)

    def handle_preemptions_with_metadata(
        self, preempted_req_ids: set[str], metadata: KVConnectorMetadata
    ) -> None:
        """Bind only this LMCache child's checkpoint controls before overwrite."""
        try:
            self.bind_connector_metadata(metadata)
            self.handle_preemptions(preempted_req_ids)
        finally:
            self.clear_connector_metadata()

    @property
    def supports_dsa_live_split_source(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_split()

    @property
    def supports_dsa_live_latent_split_source(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_latent_source()

    @property
    def supports_dsa_live_latent_split_destination(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_latent_destination()

    def configure_live_latent_source(self, enabled: bool) -> None:
        """Enable latent live-source capture after transport negotiation.

        The implementation deliberately defaults this feature to disabled so
        that upgrading LMCache without a hybrid-capable transport cannot make
        an older consumer reject the otherwise valid group-1 descriptor.
        """
        configure = getattr(
            self._lmcache_engine, "configure_live_latent_source", None
        )
        if callable(configure):
            configure(enabled)

    def _take_live_split_destination_plans(
        self, handled_groups: tuple[int, ...]
    ) -> dict[str, dict[str, Any]]:
        """Internal worker hook consumed by AscendMultiConnector."""
        return self._lmcache_engine.take_live_split_destination_plans(
            handled_groups
        )

    def _accept_live_split_results(self, results: dict[str, str]) -> None:
        """Internal worker hook for negotiated live-transfer acknowledgements."""
        self._lmcache_engine.accept_live_split_results(results)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def synchronize_staged_sfa_capture_unsafe_loads(self) -> None:
        """Finish background NPU loads before serving-time graph capture."""
        self._lmcache_engine.synchronize_staged_sfa_capture_unsafe_loads()

    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list = None,
        token_start_index: list = None,
        request_ids=None,
        target_slot_mapping=None,
        payload_event=None,
        selected_token_counts=None,
    ) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
            selected_tokens: batched per-request DSA selected token indices.
            token_start_index: per-request start offset into slot_mapping.
            request_ids: req_id for each selected_tokens row (input_batch order),
                used to pair rows to requests by identity instead of by position.
            target_slot_mapping: optional batched physical destination slots
                matching selected_tokens. Used by sparse decode scratch loads.
            payload_event: optional producer event for device selected-token
                payloads. LMCache waits on it before row selection.
        """
        self._lmcache_engine.wait_for_layer_load(
            layer_name,
            selected_tokens,
            token_start_index,
            request_ids,
            target_slot_mapping=target_slot_mapping,
            payload_event=payload_event,
            selected_token_counts=selected_token_counts,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load during the last interval."""
        return self._lmcache_engine.get_block_ids_with_load_errors()

    def get_completed_decode_window_saves(self) -> dict[str, int]:
        return self._lmcache_engine.get_completed_decode_window_saves()

    def build_connector_worker_meta(self):
        build = getattr(self._lmcache_engine, "build_connector_worker_meta", None)
        return build() if callable(build) else None

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        return self._lmcache_engine.shutdown()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        matched_tokens = self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        return matched_tokens, self._lmcache_engine.should_load_kv_async(
            request.request_id
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(
            request, num_external_tokens, blocks
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def has_pending_control(self) -> bool:
        """Report an acknowledgement waiting for an otherwise idle worker step."""
        return self._lmcache_engine.has_pending_control()

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        """
        Update scheduler-side LMCache state from worker-side connector output.
        """
        self._lmcache_engine.update_connector_output(connector_output)

    def update_connector_worker_metadata(
        self, worker_metadata: Any, active_req_ids: set[str]
    ) -> None:
        update = getattr(
            self._lmcache_engine, "update_connector_worker_metadata", None
        )
        if callable(update):
            update(worker_metadata, active_req_ids)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # LMCache's layerwise DSA path owns both groups, while its scheduler
        # completion bookkeeping is keyed by the primary group's block table.
        return self.request_finished(request, block_ids[0])


    def register_kv_caches(self, kv_caches):
        self._lmcache_engine.register_kv_caches(kv_caches)
        impl = self._lmcache_engine
        if impl._layerwise_prefill_dma:
            from vllm.v1.core.dsa_shared_pool import (
                DSASharedBlockLayout, layerwise_prefill_bundle_multiplier,
            )
            from lmcache_ascend.v1.npu_connector.layerwise_dma import (
                build_group_cycles,
                cache_page_size_bytes,
            )

            latent = impl._kvcaches_for_group(0)[0]
            indexer_layers = impl._kvcaches_for_group(1)
            # DMA cycle geometry stays in canonical BF16 logical units. Mixed
            # C8 rows use prepared packets; their INT8+scale tuple is not a raw
            # plane-width descriptor and must never define this geometry.
            indexer = next(
                (layer for layer in indexer_layers if len(layer) == 1), None
            )
            if indexer is None:
                raise ValueError(
                    "Uniform-C8 banked prefill requires explicit logical geometry"
                )
            # Match the scheduler's layout construction, including its slab split.
            layout = DSASharedBlockLayout(
                latent_page_size_bytes=cache_page_size_bytes(latent),
                indexer_page_size_bytes=cache_page_size_bytes(indexer),
                capacity_bundles=1,
                bundle_multiplier=layerwise_prefill_bundle_multiplier(),
            )
            impl.lmcache_engine.gpu_connector.prefill_dma_cycles = build_group_cycles(
                latent, indexer,
                impl._lmcache_chunk_size,
                layout.bundle_multiplier,
                (layout.k_nope_dim, layout.k_pe_dim),
            )

    @property
    def uses_layerwise_model_callbacks(self) -> bool:
        """Whether model-layer Python callbacks are part of this execution."""
        return bool(getattr(self._lmcache_engine, "use_layerwise", False))

    @property
    def supports_staged_sfa_sparse_load(self) -> bool:
        """Advertise the exact staged-SFA selective-load contract."""
        engine = self._lmcache_engine
        config = getattr(engine, "config", None)
        return bool(
            getattr(engine, "use_layerwise", False)
            and getattr(engine, "kv_role", None) in ("kv_both", "kv_consumer")
            and getattr(config, "dsa_two_groups", False)
            and getattr(config, "enable_sparse_attention", False)
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        parallel = getattr(vllm_config, "parallel_config", None)
        if transfer is not None and parallel is not None:
            extra = dict(getattr(transfer, "kv_connector_extra_config", None) or {})
            dp_rank = getattr(parallel, "data_parallel_index", None)
            if dp_rank is None:
                dp_rank = getattr(parallel, "data_parallel_rank_local", 0)
            extra["lmcache_remote_fill_destination_dp_rank"] = int(dp_rank or 0)
            extra["lmcache_remote_fill_destination_dp_size"] = int(
                getattr(parallel, "data_parallel_size", 1) or 1
            )
            transfer.kv_connector_extra_config = extra
        self._common_init(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

    def prepare_sparse_graph_step(
        self,
        target_layer_names: tuple[str, ...],
        *,
        allow_empty: bool = False,
        request_ids: tuple[str, ...] | None = None,
        frontiers: tuple[int, ...] | None = None,
    ) -> PreparedSparseSource | None | tuple[PreparedSparseSource | None, ...]:
        """Prepare and lease graph sources, leaving MTP draft loads independent."""
        return self._lmcache_engine.prepare_sparse_graph_step(
            target_layer_names,
            allow_empty=allow_empty,
            request_ids=request_ids,
            frontiers=frontiers,
        )

    def capture_live_source_event_handoff(self, forward_context: Any) -> bool:
        """Forward an armed post-forward producer event to the implementation."""

        return bool(
            self._lmcache_engine.capture_live_source_event_handoff(forward_context)
        )

    def seal_sparse_destination_layout(self) -> None:
        """Forward the final staged-capture storage contract when supported."""
        seal = getattr(self._lmcache_engine, "seal_sparse_destination_layout", None)
        if callable(seal):
            seal()

    def get_remote_fill_placement_info(
        self,
    ) -> dict[str, int | str | bool] | None:
        """Return the decoder's pointer-free remote-fill placement."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        discover = getattr(engine, "get_remote_fill_placement_info", None)
        return discover() if callable(discover) else None

    def get_remote_fill_metrics(self) -> dict[str, int] | None:
        """Return fixed-cardinality decoder protocol metrics when active."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        snapshot = getattr(engine, "get_remote_fill_metrics", None)
        return snapshot() if callable(snapshot) else None

    def remote_fill_requires_paired_restart(self) -> bool:
        """Expose an armed-transfer fatal latch to the worker supervisor."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        check = getattr(engine, "remote_fill_requires_paired_restart", None)
        return bool(check()) if callable(check) else False

    def finish_layerwise_prefill_save(self, layer_name: str) -> None:
        """Run storage publication after the layer's HCOM is submitted."""
        self._lmcache_engine.finish_layerwise_prefill_save(layer_name)

    def submit_layerwise_prefill_load(self, layer_name: str | int) -> None:
        """Submit N+2; virtual N=-1 submits L1 at first-layer entry."""
        self._lmcache_engine.submit_layerwise_prefill_load(layer_name)

    @property
    def supports_layerwise_prefill_dsa_index_transfer_window(self) -> bool:
        return (
            self._lmcache_engine.supports_layerwise_prefill_dsa_index_transfer_window
        )

    @property
    def supports_dsa_index_lmcache(self) -> bool:
        return self._lmcache_engine.supports_dsa_index_lmcache

    @property
    def supports_layerwise_prefill_transfer_window(self) -> bool:
        return self._lmcache_engine.supports_layerwise_prefill_transfer_window

    @property
    def supports_layerwise_prefill_bank_events(self) -> bool:
        """Whether this connector can fence P-node bank reuse."""
        return callable(
            getattr(
                self._lmcache_engine,
                "record_layerwise_prefill_bank_use",
                None,
            )
        )

    def record_layerwise_prefill_bank_use(self, layer_name: str, event: Any) -> None:
        """Hand off the event recorded after a P layer's final KV access.

        Args:
            layer_name: Registered latent or indexer layer name.
            event: Device event marking the layer's final bank access.
        """
        self._lmcache_engine.record_layerwise_prefill_bank_use(layer_name, event)
