import os
from typing import Dict

import torch
import torch.nn.functional as F
from gym import Space

from Model.seq2seq_policy import Seq2SeqPolicy
from Model.cma_policy import CMAPolicy
from utils.logger import logger
from src.common.param import args
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN


class VLNCETrainer:
    #
    def __init__(
        self,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        ckpt_path=None,
        policy=None,  # 新增参数，允许传入外部策略实例
    ):
        self.start_epoch = 0
        self.step_id = 0

        if not args.DistributedDataParallel:
            self.device = (
                torch.device("cuda", args.trainer_gpu_device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # 策略初始化
        if policy is not None:
            # 使用传入的自定义策略
            self.policy = policy
            self.policy.to(self.device)  # 确保在正确的设备上
        else:
            model_config = CN.clone()
            if args.policy_type == 'seq2seq':
                self.policy = Seq2SeqPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'cma':
                self.policy = CMAPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'hcm':
                self.policy = HCMPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'unet':
                self.policy = UNetPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            elif args.policy_type == 'vlnbert':
                self.policy = VLNBertPolicy.from_config(
                    observation_space=observation_space,
                    action_space=action_space,
                    out_model_config=model_config,
                    device=self.device,
                )
            else:
                raise NotImplementedError

            self.policy.to(self.device)

        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        # 优化器（始终使用传入策略的参数）
        # self.optimizer = torch.optim.Adam(
        #     self.policy.parameters(), lr=args.lr
        # )

        # 加载检查点（如果需要）
        if load_from_ckpt:
            assert os.path.isfile(ckpt_path), 'ckpt_path error'
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            self.optimizer.load_state_dict(ckpt_dict["optimizer"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        # 分布式包装
        # if args.DistributedDataParallel:
        # 仅当使用 DDP 且未使用 DeepSpeed 时才包装（DeepSpeed 会自己处理分布式）
        if args.DistributedDataParallel and not args.deepspeed:
            # 注意：如果传入的 policy 已经是 DDP 包装，此处需判断，这里假设传入的是原始模型
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    #
    def save_checkpoint(self, file_name, dagger_it, epoch) -> None:
        """
        Save checkpoint with specified name.
        :param file_name: file name for checkpoint
        :param epoch: epoch
        :return: None
        """
        checkpoint = {
            "state_dict": self.policy.module.state_dict() if args.DistributedDataParallel else self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "config": str(args),
            'dagger_it': dagger_it,
            'epoch': epoch,
        }

        from pathlib import Path
        checkpoint_folder = Path(args.project_prefix) / 'DATA/output/{}/train/checkpoint/{}'.format(args.name, args.make_dir_time)
        if not os.path.exists(str(checkpoint_folder)):
            os.makedirs(str(checkpoint_folder), exist_ok=True)

        torch.save(
            checkpoint, str(checkpoint_folder / file_name)
        )

    #
    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    #
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        # 获取原始策略（如果是 DDP 包装则取 .module）
        if not args.DistributedDataParallel:
            policy = self.policy
        else:
            policy = self.policy.module

        # if args.policy_type in ['seq2seq', 'cma']:
        #     if not args.DistributedDataParallel:
        #         recurrent_hidden_states = torch.zeros(
        #             N,
        #             self.policy.net.num_recurrent_layers,
        #             self.policy.net.state_encoder.hidden_size,
        #             device=self.device,
        #         )
        #     else:
        #         recurrent_hidden_states = torch.zeros(
        #             N,
        #             self.policy.module.net.num_recurrent_layers,
        #             self.policy.module.net.state_encoder.hidden_size,
        #             device=self.device,
        #         )
        # else:
        #     raise NotImplementedError

        # 判断策略是否有 RNN 状态（通过检查 net 属性）
        if hasattr(policy, 'net') and policy.net is not None:
            recurrent_hidden_states = torch.zeros(
                N,
                policy.net.num_recurrent_layers,
                policy.net.state_encoder.hidden_size,
                device=self.device,
            )
        else:
            # 无 RNN 状态，设为 None
            recurrent_hidden_states = None

        AuxLosses.clear()

        if not args.DistributedDataParallel:
            distribution = self.policy.build_distribution(
                observations, recurrent_hidden_states, prev_actions, not_done_masks
            )
        else:
            distribution = self.policy.module.build_distribution(
                observations, recurrent_hidden_states, prev_actions, not_done_masks
            )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

