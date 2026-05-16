"""
llama3_prompter.py

Defines a PromptBuilder for building LLaMa-3 Chat Prompts.
"""

from typing import Optional

from llava.model.base_prompter import PromptBuilder

# Default System Prompt for Prismatic Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
    "openvla": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
    "navila": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
}


def format_system_prompt(system_prompt: str) -> str:
    """为 Llama3 格式化系统提示"""
    return system_prompt.strip()


class Llama3ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None, eos=None, bos=None) -> None:
        super().__init__(model_family, system_prompt, eos, bos)

        # 格式化系统提示
        raw_system_prompt = SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        self.system_prompt = format_system_prompt(raw_system_prompt)

        # 使用 Llama3 的特殊标记
        self.bos = bos if bos is not None else "<|begin_of_text|>"
        self.eos = eos if eos is not None else "<|eot_id|>"

        # Llama3 对话格式使用特殊标记
        self.wrap_human = lambda \
            msg: f"<|start_header_id|>user<|end_header_id|>\n\n{msg}{self.eos}<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        """Add a turn to the prompt and return the added part."""
        # 移除可能的图像标记（如果需要）
        message = message.replace("<image>", "").strip()

        # 检查角色顺序（如果需要）
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # 特殊处理系统提示（turn_count == 0）
        if self.turn_count == 0 and role == "human":
            # 对于 Llama3，系统提示是独立的，不与人消息合并
            if self.system_prompt:
                system_part = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}{self.eos}"
                human_message = self.wrap_human(message)
                wrapped_message = system_part + human_message
            else:
                wrapped_message = self.wrap_human(message)
        elif role == "human":
            wrapped_message = self.wrap_human(message)
        else:  # role == "gpt"
            wrapped_message = self.wrap_gpt(message)

        # 更新提示
        self.prompt += wrapped_message

        # 增加回合计数
        self.turn_count += 1

        # 返回"wrapped_message"（添加到上下文的有效字符串）
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        """获取假设添加新用户消息后的提示（不修改当前状态）"""
        # 假设总是用户的回合！
        prompt_copy = str(self.prompt)

        # 移除可能的图像标记（如果需要）
        message = message.replace("<image>", "").strip()

        # 特殊处理系统提示（turn_count == 0）
        if self.turn_count == 0:
            # 对于 Llama3，系统提示是独立的，不与人消息合并
            if self.system_prompt:
                system_part = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}{self.eos}"
                human_message = self.wrap_human(message)
                prompt_copy = system_part + human_message
            else:
                human_message = self.wrap_human(message)
                prompt_copy = human_message
        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        # 添加 BOS token 并返回（移除前缀和右侧空白）
        full_prompt = f"{self.bos}{prompt_copy}"
        return full_prompt.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        """获取最终提示"""
        # 对于 Llama3，系统提示格式
        if self.system_prompt and self.turn_count == 0:
            system_part = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}{self.eos}"
            full_prompt = f"{self.bos}{system_part}{self.prompt}"
        else:
            full_prompt = f"{self.bos}{self.prompt}"

        # 移除前缀 <bos> 因为 tokenizer 会自动插入！
        return full_prompt.removeprefix(self.bos).rstrip()
        # return full_prompt