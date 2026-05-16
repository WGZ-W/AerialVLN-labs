# test_deepspeed.py
import torch
import deepspeed

# --- 1. 定义模型 (尽量复现你的模型结构，但可以简化)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟你的 LLaVAPolicy 的 forward 逻辑
        self.vision_encoder = torch.nn.Linear(3*384*384, 1024)
        self.action_head = torch.nn.Linear(1024, 100)

    def forward(self, pixel_values):
        # 模拟图像编码：将 [B, 3, H, W] 展平为 [B, 3*H*W]
        B, C, H, W = pixel_values.shape
        flat_pixel_values = pixel_values.view(B, -1)
        encoded = self.vision_encoder(flat_pixel_values)
        action_logits = self.action_head(encoded)
        return action_logits

# --- 2. 准备虚拟数据 (匹配你真实数据的维度)
# 你的真实 pixel_values 是 [200, 3, 384, 384]，这里用随机数据模拟
dummy_input = torch.randn(200, 3, 384, 384).cuda()

# --- 3. 初始化 DeepSpeed 引擎
model = SimpleModel().cuda()
ds_config = {
    "train_batch_size": 200,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},  # 根据你的需要开启或关闭
    "zero_optimization": {"stage": 0},  # 测试阶段可以先不用 ZeRO
}
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)

# --- 4. 执行一次前向和反向传播
print("开始测试 forward...")
output = model_engine(dummy_input)
loss = output.sum()
print(f"Forward 成功，输出形状: {output.shape}")
print("开始测试 backward...")
model_engine.backward(loss)
print("Backward 成功！")