# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch
import torch_npu  # noqa: F401

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.content_diagnostics import configure_npu_content_diagnostics
from lmcache.v1.device_connector import DeviceConnectorInterface
from lmcache.v1.device_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
)

logger = init_logger(__name__)


def CreateNPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: Optional[LayoutHints] = None,
) -> DeviceConnectorInterface:
    """Factory function to create NPU connectors on Ascend.

    The factory selects native implementations without replacing imported
    classes or modifying the process-wide torch device API.
    """
    use_gpu = need_gpu_interm_buffer(config)
    configure_npu_content_diagnostics(
        bool(getattr(config, "enable_npu_content_diagnostics", False))
    )

    num_gpus = torch.npu.device_count()
    if num_gpus <= 0:
        raise RuntimeError("An available Ascend NPU is required for KV transfer")
    local_rank = metadata.worker_id % num_gpus
    from lmcache.inference_profile import validate_device_name

    validate_device_name(torch.npu.get_device_name(local_rank))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    if engine == EngineType.VLLM:
        pass  # Unsupported P4 branch removed.

        if config.use_layerwise:
            conn = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                metadata, use_gpu, device, layout_hints=layout_hints
            )
            conn.runtime_kv_group_layer_counts = getattr(
                metadata, "runtime_kv_group_layer_counts", None
            )
            conn.dsa_two_groups = getattr(config, "dsa_two_groups", False)
            conn.enable_npu_transfer_validation = getattr(
                config, "enable_npu_transfer_validation", True
            )
            return conn

        if config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. Please contact LMCache-Ascend."
            )
        else:
            conn = VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device, layout_hints=layout_hints
            )
            conn.dsa_two_groups = getattr(config, "dsa_two_groups", False)
            return conn
    else:
        raise RuntimeError(f"Unsupported engine type for Ascend: {engine}")
