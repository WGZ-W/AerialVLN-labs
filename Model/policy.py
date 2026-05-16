from typing import Any

from torch import nn as nn
import torch

from src.common.param import args
from Model.utils.common import CategoricalNet, CustomFixedCategorical


class ILPolicy(nn.Module):
    #
    def __init__(self, net, dim_actions):
        super().__init__()

        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    #
    def forward(self, *x):
        raise NotImplementedError

    #
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        step=0,
    ):
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )

        else:
            raise NotImplementedError

        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    #
    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        else:
            raise NotImplementedError

        return self.action_distribution(features)



import torch.nn as nn
import torch.distributions as D
from llava.model.language_model.llava_llama import LlavaLlamaModel2
from llava.model.configuration_llava import LlavaConfig
from peft import LoraConfig, get_peft_model

class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"

class LLaVAPolicy(nn.Module):
    def __init__(
            self,
            vla_path,
            action_space,
            use_lora=True,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super().__init__()

        # 初始化模型
        config = LlavaLlamaConfig.from_pretrained(vla_path, resume=False)
        config.model_dtype = torch.bfloat16
        config.model_dtype = config.model_dtype.__str__()
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = vla_path
        # 加载预训练 LLaVA 模型（需提前配置 config）
        self.vlm = LlavaLlamaModel2(
            config=config,
            attn_implementation="flash_attention_2",
            model_max_length=4096,
        )
        # ).to(device)

        # 冻结所有参数
        for param in self.vlm.parameters():
            param.requires_grad = False

        if use_lora:
            # 配置 LoRA 目标模块（根据模型结构调整）
            target_modules = ["q_proj", "v_proj"]  # 常见选择，也可包括 "k_proj", "o_proj"
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()  # 打印可训练参数数量


        self.tokenizer = self.vlm.tokenizer
        self.image_processor = self.vlm.vision_tower.image_processor

        # 可选冻结部分层
        # for param in self.vlm.vision_tower.parameters():
        #     param.requires_grad = False
        # for param in self.vlm.llm.parameters():
        #     param.requires_grad = False

        # 获取 LLM 的隐藏维度
        llm_hidden_size = self.vlm.llm.config.hidden_size
        # 动作分类头
        self.action_head = nn.Linear(llm_hidden_size, action_space.n)

    def forward(self, observations, rnn_states, prev_actions, masks, **kwargs):
        # 从 observations 提取图像和指令
        # 假设 observations 包含 'pixel_values' (处理后图像) 和 'input_ids' (指令 token)
        pixel_values = observations['pixel_values']  # [batch, 3, H, W]
        input_ids = observations['input_ids']        # [batch, seq_len]

        # 历史图像特征（如果需要）
        history_values = observations.get('history_values', None)

        # 调用 LLaVA forward 得到输出
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            history_values=history_values,
            attention_mask=observations.get('attention_mask'),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        # 取最后一个 token 的隐藏状态（通常是动作预测位置）
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden]
        last_hidden = last_hidden.float()   # 转换为 float32 以匹配线性层权重
        action_logits = self.action_head(last_hidden)     # [batch, action_dim]

        # 返回与原有接口一致的 (action_logits, rnn_states)
        # 若需 RNN 状态，此处用 None 或从模型内部提取（如 past_key_values）
        return action_logits, None

    def act(
            self,
            observations,
            rnn_states,
            prev_actions,
            masks,
            deterministic=False,
            step=0,
    ):
        logits, rnn_states = self.forward(observations, rnn_states, prev_actions, masks)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        actions = actions.unsqueeze(1)  # 添加维度，变为 [batch_size, 1]
        return actions, rnn_states

    def build_distribution(self, observations, rnn_states, prev_actions, masks):
        logits, _ = self.forward(observations, rnn_states, prev_actions, masks)
        return D.Categorical(logits=logits)
