
import os
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import draccus
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, set_seed

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from llava.model.metrics import VLAMetrics
# from llava.model.strategy import TrainingStrategy
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.model import *
from llava.train.utils import (
    get_checkpoint_path,
    mprint,
    prepare_config_for_training,
    unit_test_rope_scaling,
    vision_resolution_elevation,
    get_siglip_transform,
)
from llava.model.overwatch import initialize_overwatch
from llava.dataset import get_vla_dataset_and_collator
from llava.dataset import save_dataset_statistics
from llava.model.strategy import TrainingStrategy
from llava.model.strategy_navila import SimpleFSDPWrapper



def train():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    bnb_model_from_pretrained_args = {}

    set_seed(training_args.seed)

    config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path)
    if getattr(config, "resume_path", None) is not None:
        config.resume_path = model_args.model_name_or_path

    prepare_config_for_training(config, model_args, training_args, data_args)

    # 创建模型实例
    model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )

    # 添加模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    vision_tower = model.get_vision_tower()

    if vision_tower is not None:
        # 图像处理器在这出现
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

    image_transform = get_siglip_transform(vision_tower.image_processor)

    print(image_transform)

    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_args.data_path,
        data_args.data_mix,
        # image_transform=image_transform,
        image_processor=vision_tower.image_processor,
        tokenizer=model.tokenizer,
        # default_image_resolution=model.vision_backbone.default_image_resolution,
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        image_aug=data_args.image_aug,
    )








if __name__ == "__main__":
    train()