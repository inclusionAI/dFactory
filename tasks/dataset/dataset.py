from functools import partial
from typing import Optional, Callable, Literal
from datasets import load_from_disk
from datasets.distributed import split_dataset_by_node
from pathlib import Path
from veomni.utils.dist_utils import main_process_first
from veomni.distributed.parallel_state import get_parallel_state
from veomni.data.dataset import IterativeDataset, MappingDataset


def build_local_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    source_name: Optional[str] = None,
):
    parallel_state = get_parallel_state()
    dataset = load_from_disk(Path(data_path) / namespace)
    dataset = dataset.shuffle(seed=seed)

    if transform:
        transform = partial(transform, source_name=source_name)
    return MappingDataset(dataset, transform=transform)
