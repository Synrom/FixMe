from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import Dataset

DFG = List[Tuple[str, int, str, List[str], List[int]]]
ORI2CUR = Dict[int, Tuple[int, int]]


@dataclass
class InputFeaturesTrain(object):
    """A single training features for a example."""

    source_ids: List[int]
    position_idx: List[int]
    dfg_to_dfg: List[List[int]]
    dfg_to_code: List[Tuple[int, int]]
    target_ids: List[int]
    source_mask: List[int]
    target_mask: List[int]
    target_token_length: int
    fix_token_length: int


@dataclass
class InputFeaturesTest(object):
    """A single test features for a example."""

    source_ids: List[List[int]]
    position_idx: List[List[int]]
    dfg_to_dfg: List[List[List[int]]]
    dfg_to_code: List[List[Tuple[int, int]]]
    source_mask: List[List[int]]
    decoder_context: List[int]
    target_token_length: int


# pylint: disable=E1101
TestSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

TrainSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class TextDatasetTest(Dataset):
    def __init__(
        self,
        examples: List[InputFeaturesTest],
        max_source_length: int,
        max_window_size: int,
    ):
        self.examples: List[InputFeaturesTest] = examples
        self.max_source_length: int = max_source_length
        self.max_window_size: int = max_window_size
        self.train = False

    def __len__(self):
        return len(self.examples)

    def get_desired_length(self, tensor: torch.Tensor) -> torch.Tensor:
        current_length = tensor.size(0)
        if current_length >= self.max_window_size:
            return tensor[: self.max_window_size, ...]
        repeat_times = self.max_window_size - current_length

        repeat_tuple = (repeat_times,) + (1,) * len(tensor[-1].size())

        # Repeat the last tensor along the first dimension to match the desired length
        repeated_tensor = tensor[-1].unsqueeze(0).repeat(repeat_tuple)

        # Concatenate the original tensor with the repeated tensor along the first dimension
        tensor_with_desired_length = torch.cat([tensor, repeated_tensor], dim=0)
        return tensor_with_desired_length

    def __getitem__(self, item: int) -> TestSample:
        # calculate graph-guided masked function
        windows = len(self.examples[item].source_ids)
        attn_mask = np.zeros(
            (windows, self.max_source_length, self.max_source_length),
            dtype=bool,
        )
        for window_offset in range(0, windows):
            # calculate begin index of node and max length of input
            node_index = sum(
                [i > 1 for i in self.examples[item].position_idx[window_offset]]
            )
            max_length = sum(
                [i != 1 for i in self.examples[item].position_idx[window_offset]]
            )
            # sequence can attend to sequence
            attn_mask[window_offset, :node_index, :node_index] = True
            # special tokens attend to all tokens
            for idx, i in enumerate(self.examples[item].source_ids[window_offset]):
                if i in [0, 2]:
                    attn_mask[window_offset, idx, :max_length] = True
            # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(
                self.examples[item].dfg_to_code[window_offset]
            ):
                if a < node_index and b < node_index:
                    attn_mask[window_offset, idx + node_index, a:b] = True
                    attn_mask[window_offset, a:b, idx + node_index] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(self.examples[item].dfg_to_dfg[window_offset]):
                for a in nodes:
                    if a + node_index < len(
                        self.examples[item].position_idx[window_offset]
                    ):
                        attn_mask[
                            window_offset, idx + node_index, a + node_index
                        ] = True

        return (
            self.get_desired_length(torch.tensor(self.examples[item].source_ids)),
            self.get_desired_length(torch.tensor(self.examples[item].source_mask)),
            self.get_desired_length(torch.tensor(self.examples[item].position_idx)),
            self.get_desired_length(torch.tensor(attn_mask)),
            torch.tensor(self.examples[item].decoder_context),
            torch.tensor([self.examples[item].target_token_length]),
        )


class TextDatasetTrain(Dataset):
    def __init__(
        self,
        examples: List[InputFeaturesTrain],
        max_source_length: int,
        max_window_size: int,
    ):
        self.examples: List[InputFeaturesTrain] = examples
        self.max_source_length: int = max_source_length
        self.max_window_size: int = max_window_size
        self.train = True

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item: int) -> TrainSample:
        # calculate graph-guided masked function
        attn_mask = np.zeros(
            (self.max_source_length, self.max_source_length),
            dtype=bool,
        )
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].source_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (
            torch.tensor(self.examples[item].source_ids),
            torch.tensor(self.examples[item].source_mask),
            torch.tensor(self.examples[item].position_idx),
            torch.tensor(attn_mask),
            torch.tensor(self.examples[item].target_ids),
            torch.tensor(self.examples[item].target_mask),
            torch.tensor([self.examples[item].target_token_length]),
            torch.tensor([self.examples[item].fix_token_length]),
        )
