"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from llava.model import *
# from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, training_args

from transformers.modeling_outputs import CausalLMOutputWithPast


import wandb


from llava.model.action_tokenizer import ActionTokenizer
from llava.dataset import RLDSBatchTransform, RLDSDataset
from llava.dataset.data_utils import PaddedCollatorForActionPrediction
from llava.dataset import save_dataset_statistics
from llava.training_monitor import TrainingMonitor

from keyframe.keyframe_selector import KeyframeSelector

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"

# 优化数据加载
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更好的错误信息
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path: str = "/home/weiguanzhao/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)
    # vla_path: str = "/home/weiguanzhao/navila-llama3-8b-8f"                            # Path to OpenVLA model (on HuggingFace Hub)
    vla_path: str = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    # data_root_dir: Path = Path("/mnt/sda/wgz/")        # Path to Open-X dataset directory
    # dataset_name: str = "bridge_orig"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    # run_root_dir: Path = Path("/mnt/sda/wgz/bri_fine")                               # Path to directory to store logs & checkpoints
    # adapter_tmp_dir: Path = Path("adapter-tmp")                   # Temporary directory for LoRA weights before fusing
    data_root_dir: Path = Path("/mnt/sdc/weiguanzhao/OpenFly-rlds-my")         # Path to Open-X dataset directory
    # data_root_dir: Path = Path("/mnt/sda/wgz/OpenFly-rlds")         # Path to Open-X dataset directory
    dataset_name: str = "vln_mix"                                   # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("/mnt/sdd/weiguanzhao/navila-finetune/runs-ks")  # Path to directory to store logs & checkpoints
    # run_root_dir: Path = Path("/mnt/sda/wgz/navila-finetune/runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("/mnt/sdd/weiguanzhao/navila-finetune/adapter-tmp-ks")

    # Fine-tuning Parameters
    batch_size: int = 1                                             # Fine-tuning batch size
    # batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    # max_steps: int = 100                                        # Max number of fine-tuning steps
    # save_steps: int = 1000                                          # Interval for checkpoint saving
    save_steps: int = 1000                                          # Interval for checkpoint saving
    # save_steps: int = 5000                                          # Interval for checkpoint saving

    ## 推荐搜索范围：1e-5 到 1e-3, (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-5, 5e-5)
    # learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    learning_rate: float = 1e-4                                     # Fine-tuning learning rate
    # learning_rate: float = 1e-5                                     # Fine-tuning learning rate
    # learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = False                                         # Whether to train with image augmentations
    shuffle_buffer_size: int = 10_000                               # Dataloader shuffle buffer size (can reduce if OOM)
    # shuffle_buffer_size: int = 5000                               # Dataloader shuffle buffer size (can reduce if OOM)
    # shuffle_buffer_size: int = 100_000                            # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # 续训配置
    resume_from_checkpoint: str = False  # 检查点路径，如果为空则不续训
    # resume_from_checkpoint: str = "/mnt/sdc/weiguanzhao/navila-finetune/adapter-tmp/navila+b2+lr-0.0005+lora-r32+dropout-0.0"
    # resume_from_checkpoint: str = "/mnt/sdc/weiguanzhao/navila-finetune/adapter-tmp/navila+b1+lr-0.0005+lora-r32+dropout-0.0"
    # resume_from_checkpoint: str = "/mnt/sdc/weiguanzhao/navila-finetune/adapter-tmp/navila+b2+lr-0.0005+lora-r32+dropout-0.0"  # 检查点路径，如果为空则不续训

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    # lora_rank: int = 32                                             # Rank of LoRA weight matrix
    # lora_dropout: float = 0.0                                      # Dropout applied to LoRA weights
    lora_dropout: float = 0.05                                       # Dropout applied to LoRA weights
    # weight_decay = 0.01
    # Weight Decay 数值通常比学习率小，推荐搜索范围：1e-4 到 1e-2，常见的选择包括 1e-4, 1e-3, 1e-2
    weight_decay = 1e-4
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance



    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"

    # 手动初始化分布式训练
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # local_rank = 0
    # device = torch.device(f"cuda:{local_rank}")

    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        # f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        # f"{cfg.vla_path.split('/')[-1]}"
        f"navila"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # 初始化模型
    config = LlavaLlamaConfig.from_pretrained(cfg.vla_path, resume=False)
    config.model_dtype = torch.bfloat16
    config.model_dtype = config.model_dtype.__str__()
    if getattr(config, "resume_path", None) is not None:
        config.resume_path = cfg.vla_path

    model = LlavaLlamaModel3(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=4096,
    ).to(device)

    # 启用梯度检查点
    # model.gradient_checkpointing_enable()

    # Create Action Tokenizer and Image Processor
    tokenizer = model.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer)
    image_processor = model.get_vision_tower().image_processor


    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, cfg.resume_from_checkpoint, is_trainable=True)
        print(f"Resuming from checkpoint {cfg.resume_from_checkpoint}")
        model.print_trainable_parameters()
    elif cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            # target_modules="all-linear",
            # target_modules=[
            #     "q_proj", "k_proj", "v_proj", "o_proj",
            #     "gate_proj", "up_proj", "down_proj"
            # ],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                # "x_proj",
            ],
            init_lora_weights="gaussian",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    model = DDP(
        model,
        device_ids=[local_rank],
        # output_device=local_rank,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    if cfg.resume_from_checkpoint:
        optimizer.load_state_dict(
            torch.load(f"{cfg.resume_from_checkpoint}/optimizer.pt")
        )

    # selector = KeyframeSelector(
    #     clip_model_name="ViT-B/32",
    #     detector_model_type="yolov8",
    #     threshold=0.2,  # 较低的阈值
    #     device="cpu"  # 使用CPU进行测试
    # )

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_processor=image_processor,
        # selector=selector,
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple([384, 384]),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True,
    )



    if local_rank == 0:
        print(f"The size of the dataset is {len(vla_dataset)}")

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if local_rank == 0:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        pin_memory=False,
    )

    # Initialize Logging =>> W&B
    if local_rank == 0:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    # recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    # recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    # recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_losses = deque(maxlen=100)
    recent_action_accuracies = deque(maxlen=100)
    recent_l1_losses = deque(maxlen=100)

    # Train!
    # with tqdm.tqdm(total=cfg.max_steps) as progress:
    total_steps = len(vla_dataset) // cfg.batch_size
    with tqdm.tqdm(total=total_steps) as progress:

        # 创建监控器
        monitor = TrainingMonitor(log_interval=10)

        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            batch["history_values"] = batch["history_values"].to(torch.bfloat16)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    history_values=batch["history_values"],
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # 梯度裁剪（重要！）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Compute Accuracy and L1 Loss for Logging
            # action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]

            label_size = batch["labels"].shape[1]
            action_logits = output.logits[:, -label_size : -1]
            action_preds = action_logits.argmax(dim=2)

            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            # Llama3 需要排除 eos
            eos_token_id = tokenizer.eos_token_id
            # mask = action_gt > action_tokenizer.action_token_begin_idx
            mask = (action_gt >= action_tokenizer.action_token_begin_idx) & (action_gt != eos_token_id)

            if batch_idx % 10 == 0 and local_rank == 0:
                print(f"action_preds: {action_preds[:, -9:]}")
                print(f"action_gt: {action_gt[:, -9:]}")

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # 记录到监控器
            monitor.record_step(smoothened_loss, smoothened_action_accuracy, smoothened_l1_loss, gradient_step_idx)

            # Push Metrics to W&B (every 10 gradient steps)
            # if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
            if local_rank == 0 and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )


            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                progress.update()


            # 保存检查点
            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if local_rank == 0:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    # processor.save_pretrained(run_dir)
                    model.module.save_pretrained(save_dir)
                    # 保存优化器状态
                    torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")
                    print(f"检查点已保存到: {save_dir}")
                    # model.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                try:
                    if cfg.use_lora and gradient_step_idx % 1000 == 0:
                    # if cfg.use_lora and gradient_step_idx % 5000 == 0:
                        base_vla = LlavaLlamaModel3(
                            config=config,
                            # attn_implementation="flash_attention_2",
                            model_max_length=4096,
                            device_map="cpu",
                            # model_max_length=2048,
                        )
                        # base_vla = AutoModelForVision2Seq.from_pretrained(
                        #     cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        # )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()

                        if local_rank == 0:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(str(run_dir) + "+" + str(gradient_step_idx))

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save dataset statistics to new directory
                                save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                                # Save processor and model weights to new directory
                                merged_vla.save_pretrained(str(checkpoint_dir))

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                except Exception as e:
                    print(f"Error: {e}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            # if gradient_step_idx == cfg.max_steps:
            # if gradient_step_idx == len(vla_dataset)/cfg.batch_size:
            if gradient_step_idx == total_steps:
                # print(f"Max step {cfg.max_steps} reached! Stopping training...")
                print(f"Total steps {total_steps} reached!")
                break

        # 训练结束报告
        monitor.plot_metrics(save_path="training_metrics.png")
        # if cfg.use_lora:
        #     base_vla = LlavaLlamaModel(
        #         config=config,
        #         attn_implementation="flash_attention_2",
        #         model_max_length=4096,
        #     )
        #     # base_vla = AutoModelForVision2Seq.from_pretrained(
        #     #     cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        #     # )
        #     merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        #     merged_vla = merged_vla.merge_and_unload()
        #
        #     if local_rank == 0:
        #         merged_vla.save_pretrained(str(run_dir))
        #         print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
        #
        # # Block on Main Process Checkpointing
        # dist.barrier()



if __name__ == "__main__":
    finetune()
