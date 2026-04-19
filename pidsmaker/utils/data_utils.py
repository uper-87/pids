"""Data utilities for graph batching and temporal data handling.

Provides custom PyTorch Geometric data structures and utilities for:
- Temporal graph data collation and batching
- Graph reindexing for mini-batch training
- TGN memory and neighbor sampling integration
- Multi-dataset handling and preprocessing
"""

import copy
import math
import os
import pickle
from collections import defaultdict
from functools import cached_property

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, TemporalData
from torch_geometric.data.collate import collate
from torch_geometric.data.data import size_repr
from torch_geometric.data.temporal import prepare_idx
from torch_geometric.loader import TemporalDataLoader
from torch_scatter import scatter

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.debug_tests import debug_test_batching
from pidsmaker.encoders import TGNEncoder
from pidsmaker.tgn import LastNeighborLoader
from pidsmaker.utils.dataset_utils import (
    get_node_map,
    get_num_edge_type,
    get_rel2id,
    possible_events,
)
from pidsmaker.utils.utils import get_multi_datasets, log, log_dataset_stats, log_tqdm


class CollatableTemporalData(TemporalData):
    """
    We use this class instead of TemporalData in order to easily concatenate data
    objects together without any batching behavior.
    Normal TemporalData doesn't support edge_index so we define it here.
    """

    def __init__(
        self,
        src=None,
        dst=None,
        t=None,
        msg=None,
        **kwargs,
    ):
        super().__init__(src=src, dst=dst, t=t, msg=msg, **kwargs)
        self.tgn_mode = False

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "original_edge_index":  # used to retrieve original node IDs during evaluation
            return 0
        if "edge_index" in key or key in ["src", "dst", "reindexed_original_n_id_tgn"]:
            if self.tgn_mode:
                return torch.unique(self.n_id_tgn).numel()
            return self.num_nodes
        return 0

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        return 1 if "edge_index" in key else 0

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ", ".join([size_repr(k, v) for k, v in self._store.items()])
        info += ", " + size_repr("edge_index", self.edge_index)
        return f"{cls}({info})"

    def index_select(self, idx):
        """ "Indexing to handle (2, E) index attributes"""
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.size(0) == self.num_events:
                data[key] = value[idx]
            elif value.ndim == 2 and value.size(1) == self.num_events and "index" in key:
                data[key] = value[:, idx]
        return data

    @property
    def num_nodes(self):
        if "original_n_id" in self:
            return self.original_n_id.numel()
        return self.edge_index.unique().numel()

    @cached_property
    def node_type_argmax(self):
        return self.node_type.max(dim=1).indices

    @cached_property
    def node_type_src_argmax(self):
        return self.node_type_src.max(dim=1).indices

    @cached_property
    def edge_type_argmax(self):
        return self.edge_type.max(dim=1).indices

    @cached_property
    def node_type_dst_argmax(self):
        return self.node_type_dst.max(dim=1).indices


def load_all_datasets(cfg, device, only_keep=None):
    multi_dataset = cfg.batching.multi_dataset_training
    train_data = load_data_set(cfg, split="train", multi_dataset=multi_dataset)
    val_data = load_data_set(cfg, split="val", multi_dataset=multi_dataset)
    test_data = load_data_set(cfg, split="test", multi_dataset=False)

    # Apply data sampling based on priority: explicit only_keep > data_sampling_ratio
    if only_keep is not None:
        # Priority 1: Legacy behavior - use explicit only_keep value for all datasets
        log(f"Using explicit only_keep={only_keep} for all datasets (legacy mode)")
        train_data = train_data[:only_keep]
        val_data = val_data[:only_keep]
        test_data = test_data[:only_keep]
    else:
        # Priority 2: Use data_sampling_ratio for training set only
        sampling_ratio = getattr(cfg.batching, 'data_sampling_ratio', 1.0)
        if sampling_ratio < 1.0 and sampling_ratio > 0.0:
            # Calculate sample size for training set based on sampling ratio
            # Note: load_data_set returns a list of lists when multi_dataset=True
            # For single dataset, it returns [[actual_data_list]]
            if multi_dataset:
                # Multi-dataset case: train_data is a list of data lists
                train_size = sum(len(ds) for ds in train_data)
                train_keep_per_dataset = max(1, int(train_size * sampling_ratio / len(train_data)))
                
                log(f"Data sampling enabled (multi-dataset): using {sampling_ratio*100:.0f}% of training data")
                log(f"  Total training events: {train_size}")
                log(f"  Events per dataset: {train_keep_per_dataset}")
                
                # Apply sampling to each dataset
                train_data = [ds[:train_keep_per_dataset] for ds in train_data]
            else:
                # Single dataset case: train_data is [[actual_data_list]]
                actual_train_data = train_data[0]  # Extract the inner list
                train_size = len(actual_train_data)
                train_keep = max(1, int(train_size * sampling_ratio))
                
                log(f"Data sampling enabled: using {sampling_ratio*100:.0f}% of training data")
                log(f"  Training: {train_keep}/{train_size} graphs")
                log(f"  Validation & Test: keeping full datasets for accurate evaluation")
                
                # Apply sampling to training set only (best practice for evaluation integrity)
                train_data = [actual_train_data[:train_keep]]  # Wrap back in outer list