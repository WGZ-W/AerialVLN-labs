"""
dataset.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional, Callable, Generator
from copy import deepcopy
import inspect
import numpy as np
import torch
import dlimp as dl
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info
from functools import partial
from PIL import Image
# from model import DinoSigLIPImageTransform
from llava.model.multimodal_encoder.siglip_encoder import SigLIPImageTransform
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, SiglipImageProcessor
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from llava.model.action_tokenizer import ActionTokenizer
from llava.dataset.data_utils import (
    NormalizationType,
    tree_map, 
    allocate_threads,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    decode_and_resize,
    add_pad_mask_dict,
    chunk_act_obs,
    subsample,
)

from llava.model.prompt_llama2 import LLaMa2ChatPromptBuilder
from llava.model.overwatch import initialize_overwatch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms
from torch import tensor

from llava.model.prompt_llama3 import Llama3ChatPromptBuilder

from keyframe.keyframe_selector import KeyframeSelector

overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


# 终于解决了 OOM 的问题啊，老是在GPU0上用满所有的显存
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    history_image_key: Optional[str] = None,
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = 1,
    num_parallel_calls: int = 1,
) -> Tuple[dl.DLataset, dict]:
    """

    {
        "observation": {
            "image_primary": Tensor[B, H, W, 3],      # RGB图像
            "image_wrist": Tensor[B, H, W, 3],        # 手腕相机
            "depth_primary": Tensor[B, H, W, 1],      # 深度图（可选）
            "proprio": Tensor[B, D_state],            # 本体感知状态
            "timestep": Tensor[B]                     # 时间步
        },
        "history_images": Tensor[B, 8, H, W, 3],      # RGB图像
        "task": {
            "language_instruction": Tensor[B]          # 语言指令（可选）
        },
        "action": Tensor[B, D_action],                # 动作向量
        "dataset_name": Tensor[B],                    # 数据集标识
        "absolute_action_mask": Tensor[B, D_action]   # 绝对动作掩码（可选）
    }

    将多种异构的机器人数据集转换为统一的标准化格式，为模型训练做准备。
    This function is responsible for loading a specific RLDS dataset from storage and getting it into a standardized
    format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the trajectory
    into a standard format, which includes the keys "observation" and "action". Entry "observation" should be a
    dictionary containing some number of additional keys, which will be extracted into an even more standardized format
    according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in place of an
    old name to insert padding. For example, if after `standardize_fn`, your "observation" dict has RGB images called
    "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`, then
    the resulting dataset will have an "observation" dict containing the keys "image_primary", "image_secondary", and
    "image_wrist", where "image_primary" corresponds to "workspace", "image_secondary" is a padding image, and
    "image_wrist" corresponds to "wrist".

    Entry `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which will
    be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be inserted for each
    None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will contain the
    key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset, since one
            file usually contains many trajectories)!
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to extract from the
            "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`.
            If a value of `old` is None, inserts a padding image instead (empty string).
        history_image_key (str, optional): The name of the history images to extract from the "observation" dict.
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from the
            "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding for each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key "language_instruction",
            extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory need to be made "neutral" to indicate that the task has been completed.
            For relative actions, "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - history_images                # RGB history images, present if `history_image_key` is provided
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)
    # if history_image_key is not None:
    #     REQUIRED_KEYS.add(history_image_key)

    def restructure(traj):
        # apply a standardization function, if provided
        # 首先应用可选的标准化函数处理原始轨迹
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]  # ???
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )
        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        if history_image_key is not None:
            history_images = old_obs[history_image_key]
        else:
            history_images = None

        traj = {
            "observation": new_obs,     # Dict[str, Array]
            "history_images": history_images,   # Numpy Image?
            "task": task,
            #"history": traj["history"],
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }
        
        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)
    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=1
        )

        full_dataset = full_dataset.traj_map(restructure, 1)    # ???
        
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"   # train = True
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder,
                                    split=split,
                                    shuffle=shuffle,
                                    num_parallel_reads=num_parallel_reads)

    dataset = dataset.traj_map(restructure, num_parallel_calls)
    
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of "relabeling"
    (e.g., filtering, chunking, adding goals, dropping keys).

    Transforms in this function should have the following properties:
        - They require access to an entire trajectory (i.e., they cannot be applied frame-wise).
        - They are generally not CPU-intensive, mostly involving moving and copying data.
        - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        future_action_window_size (int, optional): The number of future actions beyond window_size to include
            in the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")

        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action))

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["observation"]["proprio"]) <= max_proprio))

    # marks which entires of the observation and task dicts are padding
    # ???
    dataset = dataset.traj_map(add_pad_mask_dict, num_parallel_calls)

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions, giving them a new axis at index 1 of size `window_size` and
    # `window_size + future_action_window_size`, respectively
    dataset = dataset.traj_map(
        partial(
            chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset

def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset
    
def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a frame level. These transforms are usually more CPU-intensive, (e.g.,
    decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # Convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame["task"] = fn(frame["task"])
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        # ???!!!
        # frame["history_images"] = fn(frame["history_images"])
        return frame

    # Decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(decode_and_resize, resize_size=resize_size, depth_resize_size=depth_resize_size),
        ),
        num_parallel_calls,
    )

    return dataset


# === Core Initializer ===
def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """
    Creates an interleaved dataset from list of dataset configs (kwargs). Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overridden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overridden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            dataset according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            dataset according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # Default to uniform sampling (if `sample_weights` is not specified)
    # 权重处理, 默认均匀采样，长度校验。
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

    # Check valid `traj_transform_kwargs` and `frame_transform_kwargs`
    # transform kwargs 校验???
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    # Get Dataset Sizes
    # 获取数据集大小（关键步骤）, 用于后续权重平衡和长度估计。
    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        # "dataset_frame_transform_kwargs" is None
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        # Tuple[dl.DLataset, dict]
        _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # Get the indices of the "primary" dataset (i.e., dataset with sample_weight == 1.0)
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])

    # Balance and Normalize Weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
    #   =>> Note :: Only counting the "primary" dataset (i.e., dataset with sample_weight == 1.0)
    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

    # Allocate Threads based on Weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info("Threads per Dataset: %s", threads_per_dataset)
    overwatch.info("Reads per Dataset: %s", reads_per_dataset)

    # Construct Datasets
    overwatch.info("Constructing dataset...")
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs")
            if "dataset_frame_transform_kwargs" in dataset_kwargs
            else {}
        )
        #
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        #
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        #
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    # Interleave at the Frame Level
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # Validation =>> fix a single shuffle buffer of data and cache it in RAM; prevents gradual memory increase!
    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()

    # Shuffle the Dataset
    #   =>> IMPORTANT :: Shuffle AFTER .cache(), or else memory will still leak!
    # 添加 seed 保证可重复性
    dataset = dataset.shuffle(shuffle_buffer_size, seed=42)

    # Apply Frame Transforms
    overwatch.info("Applying frame transforms on dataset...")
    # ???
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # [Contract] When training VLA Policies, we let the Collator handle Batching!
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Note =>> Seems to reduce memory usage without affecting speed?
    dataset = dataset.with_ram_budget(1)

    # Save for Later
    dataset.sample_weights = sample_weights

    return dataset, dataset_len, all_dataset_statistics

def vln_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
   
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :3],
            trajectory["action"][:, 3:],
        ],
        axis=1,
    )
    return trajectory

def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    # load_camera_views: Tuple[str] = ("primary",),
    load_camera_views: Tuple[str, ...] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    load_history: bool = False,
    # history_length: int = 8,  # 最大8帧
    # history_stride: int = 1,  # 帧间步长

) -> Dict[str, Any]:
    """
    Generates config (kwargs) for given dataset from Open-X Embodiment.

    return: {
        name: str,
        data_dir: str,
        image_obs_keys: Dict[str, str],
        history_image_key: str,
        language_key: str,
        standardize_fn: Dict[str, Any],
        }
    """
    dataset_kwargs = {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": "image_3",},
        "history_image_key": "history_images",  # 历史帧字段名
        # "history_length": history_length,
        # "history_stride": history_stride,
        # "history_reverse": False,  # 假设最近帧在最后（索引7）
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_encoding": 2,
        "action_encoding": 5,
    }

    # Adjust Loaded Camera Views
    # 相机视图校验逻辑, 确保请求的视图在当前配置中存在
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    # 视图过滤, 只保留用户请求的视图
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # 如果不加载历史，移除相关配置
    if not load_history:
        dataset_kwargs.pop("history_image_key", None)
        # dataset_kwargs.pop("history_length", None)
        # dataset_kwargs.pop("history_stride", None)
        # dataset_kwargs.pop("history_reverse", None)
    # else:
        # 确保请求的历史长度不超过可用长度
        # if history_length > 8:
        #     overwatch.warning(f"Requested history_length={history_length} exceeds maximum 8. Clamping to 8.")
        #     dataset_kwargs["history_length"] = 8

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    dataset_kwargs["standardize_fn"] = vln_transform

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    # load_camera_views: Tuple[str] = ("primary",),
    load_camera_views: Tuple[str, ...] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    load_history: bool = False,
) -> Tuple[List[Dict[str, Any]], List[float]]:
# ) -> Tuple[Dict[str, Any], List[float]]:

    """
    根据指定的数据集混合规范，生成多个数据集的配置参数和采样权重，用于构建交错采样(interleaved)的数据集。
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted dataset (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.
    :param load_history: Load history information.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    # 去重逻辑, 防止同一个数据集被多次加入混合（避免重复采样或配置冲突）
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    # 构建 per-dataset 配置
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(
                # Dict
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                    load_history,
                )
            )
            sampling_weights.append(d_weight)

        except ValueError as e:
            overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")

    return per_dataset_kwargs, sampling_weights

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_processor: SiglipImageProcessor
    selector: KeyframeSelector = None
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img_cur = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        history_images = [Image.fromarray(rlds_batch["history_images"][image_idx]) for image_idx in range(len(rlds_batch["history_images"]))]
        img_past1 = Image.fromarray(rlds_batch["observation"]["image_secondary"][0])
        img_past2 = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        prompt_builder = Llama3ChatPromptBuilder("navila")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # selector = KeyframeSelector(
        #     clip_model_name="ViT-B/32",
        #     detector_model_type="yolov8",
        #     threshold=0.2,  # 较低的阈值
        #     device="cpu"  # 使用CPU进行测试
        # )
        # result = self.selector.process_images(history_images, lang)

        # 从result获取关键帧索引
        # keyframe_indices = result["keyframe_indices"]

        # 提取关键帧
        # selected_images = [history_images[idx] for idx in keyframe_indices]

        # 处理历史帧和关键帧
        history_image_transformer = Compose(
            [
                Resize(size=(384, 384), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                       max_size=None, antialias=True),
                CenterCrop(size=(384, 384)),
                ToTensor(),
                Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
            ]
        )
        # cur_img = image_transformer2(img_cur)

        #
        # tr_img_cur, tr_pst1, tr_pst2 = self.image_transform(img_cur), self.image_transform(img_past1), self.image_transform(img_past2)# , self.image_transform(img_past3),

        # Train ...
        # pixel_values = {
        #         k: torch.cat(
        #               (tr_img_cur[k], tr_pst1[k], tr_pst2[k]), dim=0  # , tr_pst3[k]
        #         )
        #         for k in tr_img_cur.keys()
        #     }

        # Finetune ...
        # pixel_values = tr_img_cur
        # print(f"Images resolution is {img_cur.size}")

        pixel_values = self.image_processor(img_cur, return_tensors="pt").pixel_values
        ## KS
        tr_pst1 = self.image_processor(img_past1, return_tensors="pt").pixel_values
        tr_pst2 = self.image_processor(img_past2, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)
        tr_pst1 = tr_pst1.squeeze(0)
        tr_pst2 = tr_pst2.squeeze(0)

        ## Random KS
        # index1 = random.randint(0, 7)
        # index2 = random.randint(0, 7)
        # ran_ks_1 = history_images[index1]
        # ran_ks_2 = history_images[index2]
        # ran_ks_1 = self.image_processor(ran_ks_1, return_tensors="pt").pixel_values
        # ran_ks_2 = self.image_processor(ran_ks_2, return_tensors="pt").pixel_values
        # ran_ks_1 = ran_ks_1.squeeze(0)
        # ran_ks_2 = ran_ks_2.squeeze(0)

        # History information
        history_values = []
        # for x in history_images:
        ## Prompt Select Images
        # for x in selected_images:
        #     x = history_image_transformer(x)
        #     history_values.append(x)

        ## KS
        history_values.append(tr_pst1)
        history_values.append(tr_pst2)

        ## Random KS
        # history_values.append(ran_ks_1)
        # history_values.append(ran_ks_2)

        history_values = torch.stack(history_values, dim=0)

        # Mamba pos embed
        # current_len = history_values.size(0)
        # target_len = 4
        # if current_len < target_len:
        #     # 长度不足：在前面补零
        #     pad_len = target_len - current_len
        #     pad_shape = (pad_len,) + history_values.shape[1:]  # 保持其余维度一致
        #     pad_tensor = torch.zeros(pad_shape, dtype=history_values.dtype, device=history_values.device)
        #     history_values = torch.cat([pad_tensor, history_values], dim=0)
        #
        # elif current_len > target_len:
        #     # 长度超出：只保留最后 target_len 个
        #     history_values = history_values[-target_len:]


        # history_values = history_values.unsqueeze(0)
        # history_values = self.image_processor(history_images, return_tensors="pt").history_images
        # print(f"pixel_values.shape is {pixel_values.shape}")
            
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
        # 添加文本，用来 Mamba 处理
        # return dict(pixel_values=pixel_values, language=lang, cur_img=cur_img,

        return dict(
            pixel_values=pixel_values,
            labels=labels,
            input_ids=input_ids,
            history_values=history_values,
        )
        return dict(pixel_values=pixel_values, language=lang,
                    input_ids=input_ids, labels=labels,
                    dataset_name=dataset_name)




class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix    # 'vln_mix'
        self.batch_transform = batch_transform

        OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {

            # "vln_mix" : [("vlnv" + str(idx), 1.0) for idx in range(1, 21)],
            "vln_mix" : [
                # ("vln_norm", 1.0),
                # ("vlnv1", 1.0),
                # ("vlnv2", 1.0),
                # ("vlnv3", 1.0),
                # ("vlnv4", 1.0),
                # ("vlnv5", 1.0),
                # ("vlnv6", 1.0),
                # ("vln", 1.0),
                ("vln_history", 1.0),
            ],
        }

        mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]    # [('vln', 1.0)]


        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views= ("primary", "secondary", "wrist",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            # Change to True if using history
            # load_history=False,
            load_history=True,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update(
                {
                    "image_augment_kwargs" : dict(
                        random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.05],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    )
                }
            ),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    # def __iter__(self) -> Dict[str, Any]:
    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)


    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
