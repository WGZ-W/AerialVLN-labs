
import os
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import draccus
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, set_seed

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


# 这样所有 wandb.init() 调用会被跳过，训练照常跑，但不会上传日志。
os.environ["WANDB_MODE"] = "disabled"

# 明确指定你只有 2 卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def train():

    global local_rank

    # 初始化分布式
    # dist.init_process_group(backend="nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)

    # print(f"Global rank {global_rank}, local rank {local_rank} using GPU {local_rank}")

    # 在所有rank上同步并打印信息
    # dist.barrier()
    # if dist.get_rank() == 0:
    #     print("=== Distributed Training Info ===")
    #     print(f"Total processes: {dist.get_world_size()}")
    #     print("All processes should be using different GPUs")

    # 每个进程报告自己使用的GPU
    # print(
    #     f"Rank {dist.get_rank()}: Using GPU {torch.cuda.current_device()}, Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize Overwatch =>> Wraps `logging.Logger`
    overwatch = initialize_overwatch(__name__)

    # 多卡训练使用 local_rank(), 单卡训练使用 rank()
    # torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.set_device(device_id := overwatch.rank())
    torch.cuda.empty_cache()

    run_id = (
        f"n{training_args.expected_world_size // 8}+b{training_args.per_device_train_batch_size}+x{training_args.seed}"
    )

    # os.makedirs(run_dir := (data_args.run_root_dir / run_id), exist_ok=True)
    # os.makedirs(data_args.run_root_dir / run_id / "checkpoints", exist_ok=True)

    worker_init_fn = None

    # FIXME(zhijianl): This should be deprecated when we move to the new scripts.
    if os.getenv("RUN_NAME") is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]

    local_rank = training_args.local_rank
    # 支持多种精度训练
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    from transformers import BitsAndBytesConfig

    # bnb_model_from_pretrained_args.update(
    #     dict(
    #         device_map={"": training_args.device},
    #         load_in_4bit=training_args.bits == 4,
    #         load_in_8bit=training_args.bits == 8,
    #         quantization_config=BitsAndBytesConfig(
    #             load_in_4bit=training_args.bits == 4,
    #             load_in_8bit=training_args.bits == 8,
    #             llm_int8_skip_modules=["mm_projector"],  # 跳过多模态投影器的量化
    #             llm_int8_threshold=6.0,
    #             llm_int8_has_fp16_weight=False,
    #             bnb_4bit_compute_dtype=compute_dtype,
    #             bnb_4bit_use_double_quant=training_args.double_quant,
    #             bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
    #         ),
    #     )
    # )

    set_seed(training_args.seed)

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    # 根据检查点恢复训练
    if resume_path:
        resume_from_checkpoint = True
        model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)
        config.resume_path = model_args.model_name_or_path
    else:
        # First time training
        # 首次训练，根据模型名称选择对应的配置和模型类
        resume_from_checkpoint = False
        ## llm and default multimodal model
        # 默认使用LLaMA架构
        model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    ## extra configurations
    # 准备训练配置
    prepare_config_for_training(config, model_args, training_args, data_args)

    # 创建模型实例
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )

    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    mprint(model)

    # model.get_llm().requires_grad_(training_args.tune_language_model)
    model.get_llm().requires_grad_(False)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        # model.get_vision_tower().requires_grad_(False)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        mprint(f"vision tower {training_args.tune_vision_tower}")
        mprint(f"mm projector {training_args.tune_mm_projector}")

    # 添加模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    model.llm.config.use_cache = False
    model.llm.config.gradient_checkpointing_enable = True

    # Take a look on model architecture.
    # mprint(model)


    if not any(
            [training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]
    ):
        logging.warning("You are not tuning any part of the model. Please check if this is intended.")

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
        image_processor=vision_tower.image_processor,
        tokenizer=model.tokenizer,
        # predict_stop_token=True,
        # default_image_resolution=tuple([384, 384]),
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        image_aug=data_args.image_aug,
    )

    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, training_args.run_dir)



    stage = "full-finetune"

    train_strategy = TrainingStrategy(
        vlm=model,
        device_id=device_id,
        stage=stage,
        epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        global_batch_size=training_args.global_batch_size,
        per_device_batch_size=training_args.per_device_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        enable_gradient_checkpointing=training_args.enable_gradient_checkpointing,
        enable_mixed_precision_training=training_args.enable_mixed_precision_training,
        reduce_in_full_precision=training_args.reduce_in_full_precision,
        mixed_precision_dtype=training_args.mixed_precision_dtype,
        worker_init_fn=worker_init_fn,
        sharding_strategy="full-shard",
    )

    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    metrics = VLAMetrics(
        data_args.trackers,
        run_id,
        run_dir,
        draccus.encode(data_args),
        wandb_project=data_args.wandb_project,
        wandb_entity=data_args.wandb_entity,
        resume_step=data_args.resume_step,
        resume_epoch=data_args.resume_epoch,
    )

    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=data_args.save_interval,
        history_frames=training_args.history_frames,
        grid_size=training_args.grid_size,
    )

    metrics.finalize()
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    train()