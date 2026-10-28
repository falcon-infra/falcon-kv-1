# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import asyncio
import importlib

# Third Party
import torch
import torch_npu  # noqa: F401

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.startup_trace import startup_phase
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.npu_p2p_backend import AscendP2PBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def is_npu_worker(metadata: LMCacheMetadata) -> bool:
    """
    Check if the current role is a worker and an Ascend NPU is available.

    Args:
        metadata: The LMCache engine metadata.

    Returns:
        True if the worker is not a scheduler and an Ascend NPU is available.
    """
    return metadata.role != "scheduler" and torch.npu.is_available()


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    loop: asyncio.AbstractEventLoop,
    dst_device: str = "npu",
    lmcache_worker: Optional["LMCacheWorker"] = None,  # noqa: F821
    skip_backends=None,
    existing_backends=None,
) -> OrderedDict[str, StorageBackendInterface]:
    if is_npu_worker(metadata):
        dst_device = f"npu:{torch.npu.current_device()}"
    else:
        dst_device = "cpu"
    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()

    if config.enable_pd:
        # First Party
        from lmcache.v1.storage_backend.pd import AscendPDBackend

        if config.use_layerwise:
            raise ValueError(
                "Invalid LMCache-Ascend config: `enable_pd=true` is not compatible "
                "with `use_layerwise=true`. PD backend does not support layerwise "
                "mode (including pull/delay-pull paths). Disable one of them."
            )
        with startup_phase("backend_pd", rank=metadata.worker_id):
            storage_backends["PDBackend"] = AscendPDBackend(config, metadata)

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    local_cpu_backend: Optional[LocalCPUBackend] = None
    if metadata.role == "scheduler":
        # For scheduler role, local_cpu_backend is None
        pass
    elif not config.enable_pd or config.local_cpu:
        if config.max_local_cpu_size > 0:
            with startup_phase("backend_cpu", rank=metadata.worker_id):
                local_cpu_backend = LocalCPUBackend(
                    config,
                    metadata,
                    dst_device,
                    lmcache_worker,
                )
            backend_name = str(local_cpu_backend)
            storage_backends[backend_name] = local_cpu_backend
        else:
            logger.info("No cpu memory is allocated as max_local_cpu_size <= 0")

    if config.enable_p2p:
        if config.use_layerwise:
            raise ValueError(
                "Invalid LMCache-Ascend config: `enable_p2p=true` is not compatible "
                "with `use_layerwise=true`. The Ascend P2P backend does not support "
                "layerwise mode in current implementation. Disable one of them."
            )
        assert local_cpu_backend is not None
        assert lmcache_worker is not None
        p2p_backend = AscendP2PBackend(
            config,
            metadata,
            loop,
            local_cpu_backend,
            lmcache_worker,
        )
        backend_name = str(p2p_backend)
        storage_backends[backend_name] = p2p_backend

    if config.local_disk and config.max_local_disk_size > 0:
        assert local_cpu_backend is not None
        local_disk_backend = LocalDiskBackend(
            config, loop, local_cpu_backend, dst_device, lmcache_worker, metadata
        )

        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        with startup_phase("backend_remote", rank=metadata.worker_id):
            remote_backend = RemoteBackend(
                config,
                metadata,
                loop,
                local_cpu_backend,
                dst_device,
            )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    if not config.enable_pd or config.local_cpu:
        # Load storage backends from configuration
        storage_plugin_launcher(
            config,
            metadata,
            loop,
            local_cpu_backend,
            dst_device,
            storage_backends,
        )

    # Only wrap if audit is enabled in config
    if config.extra_config is not None and config.extra_config.get(
        "audit_backend_enabled", False
    ):
        # Third Party
        from lmcache.v1.storage_backend.audit_backend import AuditBackend

        # Conditionally wrap backends with audit logging if enabled in config
        audited_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
        for name, backend in storage_backends.items():
            # Wrap each normal backend with AuditBackend
            if not isinstance(backend, LocalCPUBackend):
                audited_backend = AuditBackend(backend)
                audited_backends[name] = audited_backend
                logger.info(f"Wrapped {name} with AuditBackend")
            else:
                audited_backends[name] = backend
                logger.info(f"Do not wrap {name} as it is a LocalCPUBackend")
        return audited_backends
    else:
        # If audit is not enabled, use the original backends
        return storage_backends


def storage_plugin_launcher(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: Optional[LocalCPUBackend],
    dst_device: str,
    storage_backends: OrderedDict[str, StorageBackendInterface],
) -> None:
    """
    Loads custom storage backends based on configuration.

    Looks for backend configurations in config.extra_config and instantiates
    them using the specified module and class names.
    """
    # Get the list of allowed external backends if configured
    storage_plugins = set(config.storage_plugins) if config.storage_plugins else set()
    if storage_plugins and not config.extra_config:
        logger.warning(
            "storage_plugins=%s is set but extra_config is empty; "
            "plugin settings must be provided under extra_config, e.g. "
            "extra_config.storage_plugin.<name>.module_path/class_name",
            sorted(storage_plugins),
        )
        return
    if not config.extra_config:
        return

    for storage_plugin in storage_plugins:
        try:
            module_path = config.extra_config.get(
                f"storage_plugin.{storage_plugin}.module_path"
            )
            class_name = config.extra_config.get(
                f"storage_plugin.{storage_plugin}.class_name"
            )

            if not module_path or not class_name:
                logger.warning(
                    f"Backend {storage_plugin} missing module_path or class_name"
                )
                continue

            logger.warning(
                "The 'memory_allocator' argument is deprecated and will "
                "be ignored. Storage backends now manage their own memory "
                "allocators since PR "
                "https://github.com/LMCache/LMCache/pull/1578"
            )

            # Dynamically import the module
            module = importlib.import_module(module_path)
            # Get the class from the module
            backend_class = getattr(module, class_name)

            # Create the backend instance
            backend_instance = backend_class(
                config=config,
                dst_device=dst_device,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=loop,
            )

            # Add to storage backends
            storage_backends[storage_plugin] = backend_instance
            logger.info(f"Created dynamic backend: {storage_plugin}")

        except Exception as e:
            logger.error(f"Failed to create backend {storage_plugin}: {str(e)}")
