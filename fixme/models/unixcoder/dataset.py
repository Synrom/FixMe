from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import Dataset


@dataclass
class InputFeaturesTrain(object):
    """A single training features for a example."""

    source_ids: List[int]
    target_ids: List[int]
    target_token_length: int
    fix_token_length: int
    lang: str = ""


@dataclass
class InputFeaturesTest(object):
    """A single test features for a example."""

    source_ids: List[List[int]]
    decoder_context: List[int]
    target_token_length: int


# pylint: disable=E1101
TestSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

TrainSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
]


class TextDatasetTest(Dataset):
    def __init__(
        self,
        examples: List[InputFeaturesTest],
        max_window_size: int,
    ):
        self.examples: List[InputFeaturesTest] = examples
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
        return (
            self.get_desired_length(torch.tensor(self.examples[item].source_ids)),
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
        return (
            torch.tensor(self.examples[item].source_ids),
            torch.tensor(self.examples[item].target_ids),
            torch.tensor([self.examples[item].target_token_length]),
            torch.tensor([self.examples[item].fix_token_length]),
            self.examples[item].lang,
        )

