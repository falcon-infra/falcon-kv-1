# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.indexer_c8 import IndexerC8Layout

logger = init_logger(__name__)


def validate_two_group_layer_counts(
    counts: Optional[Sequence[int]],
) -> tuple[int, int]:
    """Validate serving-engine runtime cardinalities for DSA groups.

    Args:
        counts: Runtime layer counts in latent/indexer order.

    Returns:
        The validated latent and indexer layer counts.

    Raises:
        ValueError: If runtime metadata is missing or does not describe
            exactly two non-empty groups.
    """
    if counts is None:
        raise ValueError(
            "DSA two-group mode requires runtime KV group layer counts from "
            "the serving engine"
        )
    if len(counts) == 2 and all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in counts
    ):
        return int(counts[0]), int(counts[1])
    raise ValueError(
        "DSA two-group mode requires exactly two positive runtime layer "
        f"counts, got: {counts!r}"
    )


def resolve_kv_group_num_layers(
    *,
    kv_group: int,
    dsa_two_groups: bool,
    model_num_layers: int,
    registered_groups: Sequence["KVLayerGroupInfo"],
    runtime: Optional[Sequence[int]] = None,
) -> int:
    """Resolve the authoritative transfer cardinality of a KV group.

    Shared by the cache engine and standalone lookup clients. Resolution
    order (fail-closed on disagreement):

    DSA mode requires serving-engine runtime counts. Registered worker groups
    are then validated against that expected topology before being used.

    Args:
        kv_group: The KV group index (0 = latent, 1 = indexer).
        dsa_two_groups: Whether DSA two-group mode is enabled.
        model_num_layers: The model forward layer count.
        registered_groups: Registered per-group layer info, if any.
        runtime: Serving-engine-resolved per-group layer counts, if any.

    Returns:
        The number of layers in the group.

    Raises:
        ValueError: If runtime metadata is missing or invalid, kv_group is out
            of range, or registered cardinalities disagree.
    """
    if not dsa_two_groups:
        return model_num_layers
    runtime_counts = validate_two_group_layer_counts(runtime)
    if registered_groups:
        if len(registered_groups) != 2:
            raise ValueError(
                "DSA two-group mode requires exactly two registered KV layer "
                f"groups, got {len(registered_groups)}"
            )
        if kv_group < 0 or kv_group >= len(registered_groups):
            raise ValueError(
                "KV group is out of range for registered layer groups: "
                f"kv_group={kv_group}, num_groups={len(registered_groups)}"
            )
        registered = tuple(group.num_layers for group in registered_groups)
        if registered != runtime_counts:
            raise ValueError(
                "Runtime KV group metadata disagrees with the registered KV "
                f"layer groups: runtime={list(runtime_counts)}, "
                f"registered={list(registered)}."
            )
        return registered[kv_group]
    if kv_group < 0 or kv_group >= len(runtime_counts):
        raise ValueError(
            "KV group is out of range for runtime metadata: "
            f"kv_group={kv_group}, group_counts={list(runtime_counts)}"
        )
    return runtime_counts[kv_group]


@dataclass
class KVLayerGroupInfo:
    """Information about a group of layers with the same KV cache structure.

    Layers within the same group have identical shape and dtype for their KV cache.
    Different groups may have different shapes (especially head_size) and/or dtypes.
    """

    """ List of layer names belonging to this group """
    layer_names: list[str]
    """ List of layer indices (0-based) belonging to this group """
    layer_indices: list[int]
    """ Shape of the KV cache tensor for layers in this group """
    """ For MHA: typically [2, num_blocks, block_size, num_heads, head_size] """
    """ For MLA: typically [num_blocks, block_size, head_size] """
    shape: torch.Size
    """ Data type of the KV cache tensor for layers in this group """
    dtype: torch.dtype

    # Internal sets for fast membership checking
    _layer_indices_set: set[int] = field(init=False, repr=False)
    _layer_names_set: set[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize sets for fast membership checking."""
        self._layer_indices_set = set(self.layer_indices)
        self._layer_names_set = set(self.layer_names)

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_names)}, "
            f"indices={indices_repr}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )

    @property
    def num_layers(self) -> int:
        """Return the number of layers in this group."""
        return len(self.layer_names)

    @property
    def hidden_dim_size(self) -> int:
        """Return the size of the hidden dimension in this group."""
        # hidden_dim_size = num_heads * head_size
        if len(self.shape) == 5:
            # MHA
            return self.shape[3] * self.shape[4]
        elif len(self.shape) == 4:
            # NOTE(gingfung): Ascend separated format for KVCaches
            # i.e. a tuple of kv (numblocks, blocksize, heads, headdim)
            #      very unlikely, but potentially MLA with (1, ....)
            if self.shape[0] == 1:
                raise ValueError(f"Invalid shape for hidden dim size: {self.shape}")

            return self.shape[2] * self.shape[3]
        elif len(self.shape) == 3:
            # MLA
            return self.shape[2]
        else:
            raise ValueError(f"Invalid shape: {self.shape}")

    def contains_layer(self, layer_idx: int) -> bool:
        """Check if a layer index belongs to this group."""
        return layer_idx in self._layer_indices_set

    def contains_layer_name(self, layer_name: str) -> bool:
        """Check if a layer name belongs to this group."""
        return layer_name in self._layer_names_set


@dataclass
class KVLayerGroupsManager:
    """Manager for KV layer groups with the same structure.

    This class encapsulates the functionality for managing groups of layers
    that have identical KV cache structure (shape and dtype).
    """

    kv_layer_groups: list[KVLayerGroupInfo] = field(default_factory=list)

    @property
    def num_groups(self) -> int:
        """Return the number of KV layer groups."""
        return len(self.kv_layer_groups)

    def get_group_by_layer_idx(self, layer_idx: int) -> Optional[KVLayerGroupInfo]:
        """Get the KVLayerGroupInfo for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The KVLayerGroupInfo containing this layer, or None if not found.
        """
        for group in self.kv_layer_groups:
            if group.contains_layer(layer_idx):
                return group
        return None

    def get_group_by_layer_name(self, layer_name: str) -> Optional[KVLayerGroupInfo]:
        """Get the KVLayerGroupInfo for a given layer name.

        Args:
            layer_name: The name of the layer.

        Returns:
            The KVLayerGroupInfo containing this layer, or None if not found.
        """
        for group in self.kv_layer_groups:
            if group.contains_layer_name(layer_name):
                return group
        return None

    def get_layer_shape(self, layer_idx: int) -> Optional[torch.Size]:
        """Get the shape of the KV cache for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The shape, or None if layer not found.
        """
        group = self.get_group_by_layer_idx(layer_idx)
        return group.shape if group else None

    def get_layer_dtype(self, layer_idx: int) -> Optional[torch.dtype]:
        """Get the dtype of the KV cache for a given layer index.

        Args:
            layer_idx: The 0-based index of the layer.

        Returns:
            The dtype, or None if layer not found.
        """
        group = self.get_group_by_layer_idx(layer_idx)
        return group.dtype if group else None

    def build_kv_layer_groups(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build KV layer groups structure by analyzing each layer's shape and dtype.

        Layers with the same shape and dtype are grouped together. This is useful
        because different layers may have different structures (especially the
        last dimension head_size may differ between groups), and different groups
        may have different dtypes.

        If layer groups are already built (non-empty list), this method does nothing.

        Args:
            kv_caches: Dictionary mapping layer names to KV cache tensors.
        """
        # Skip if already built (non-empty list)
        if len(self.kv_layer_groups) > 0:
            return

        if len(kv_caches) == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        # Group layers by (shape, dtype) in a single loop
        groups_dict: dict[tuple[object, ...], list[tuple[str, int]]] = defaultdict(list)
        group_infos: dict[tuple[object, ...], tuple[torch.Size, torch.dtype]] = {}

        for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
            key, shape, dtype = _get_kv_cache_group_key_and_info(kv_cache)
            groups_dict[key].append((layer_name, idx))
            group_infos[key] = (shape, dtype)

        # Build KVLayerGroupInfo list
        # Sort groups by the first layer index to maintain order
        def _get_first_layer_index(shape_dtype_key):
            """Get the index of the first layer in a layer group."""
            layer_group = groups_dict[
                shape_dtype_key
            ]  # list of (layer_name, layer_index) tuples
            first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
            layer_index = first_layer_info[1]  # extract the layer index
            return layer_index

        sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

        kv_layer_groups: list[KVLayerGroupInfo] = []
        for key in sorted_keys:
            shape, dtype = group_infos[key]
            layers = groups_dict[key]
            layer_names, layer_indices = zip(*layers, strict=False)

            group_info = KVLayerGroupInfo(
                layer_names=list(layer_names),
                layer_indices=list(layer_indices),
                shape=shape,
                dtype=dtype,
            )
            kv_layer_groups.append(group_info)

        # Store the built groups
        self.kv_layer_groups = kv_layer_groups

        # Print the group structure
        logger.info("KV layer groups: %s", kv_layer_groups)


def _get_tuple_storage_shape(kv_cache: tuple[torch.Tensor, ...]) -> torch.Size:
    """Return the flattened LMCache storage shape for tuple-based KV caches.

    For MLA/DSA, LMCache stores multiple KV tensors as a single contiguous
    hidden dimension, so we must derive the flattened hidden size from the
    whole tuple instead of only looking at the first tensor.
    """
    first_shape = kv_cache[0].shape

    for tensor in kv_cache[1:]:
        if tensor.shape[:2] != first_shape[:2]:
            raise ValueError(
                "All KV tensors in a tuple must share [num_blocks, block_size], "
                f"got {first_shape} and {tensor.shape}"
            )

    if len(kv_cache) == 2 and kv_cache[0].shape == kv_cache[1].shape:
        return first_shape

    total_hidden_dim = sum(tensor.shape[-2] * tensor.shape[-1] for tensor in kv_cache)
    return torch.Size([first_shape[0], first_shape[1], total_hidden_dim])


def _get_kv_cache_group_key_and_info(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[tuple[object, ...], torch.Size, torch.dtype]:
    """Build a stable grouping key plus the LMCache storage shape/dtype."""
    if isinstance(kv_cache, tuple):
        dtypes = tuple(tensor.dtype for tensor in kv_cache)
        if dtypes == (torch.int8, torch.float16):
            key, scale = kv_cache
            if (
                key.ndim != 4 or key.shape[1:3] != (128, 1)
                or scale.shape != (*key.shape[:3], 1)
            ):
                raise ValueError(
                    "Indexer C8 requires matching PA_BSND key/scale planes"
                )
            layout = IndexerC8Layout(key.shape[-1])
            shapes = tuple(tensor.shape for tensor in kv_cache)
            return (
                (shapes, dtypes),
                torch.Size([key.shape[0], key.shape[1], layout.token_bytes]),
                torch.uint8,
            )
        if len(set(dtypes)) != 1:
            raise ValueError(
                "Tuple-based KV caches with mixed dtypes are not supported by "
                "LMCache-Ascend."
            )

        shapes = tuple(tensor.shape for tensor in kv_cache)
        storage_shape = _get_tuple_storage_shape(kv_cache)
        return (shapes, dtypes), storage_shape, dtypes[0]

    if isinstance(kv_cache, torch.Tensor):
        return ((kv_cache.shape,), (kv_cache.dtype,)), kv_cache.shape, kv_cache.dtype

    raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")
