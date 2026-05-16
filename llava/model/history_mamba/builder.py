import os

# import VideoMamba.videomamba.video_sm.models.videomamba as videomamba
import torch
from transformers import PretrainedConfig

from llava.model.history_mamba.history_mamba import VisionMamba
# from timm.models.vision_transformer import _cfg
# from llava.model.history_mamba.video_mamba import HistoryMamba
# from .video_mamba import HistoryMamba


# def build_history_mamba(root_path: str, pretrained=True, **kwargs):
#     if root_path is None:
#         return None
#     model_name_or_path = os.path.join(root_path, "history_mamba")
#     model_name_or_path = os.path.join(model_name_or_path, "videomamba_m16_in1k_res224.pth")
#
#     model: HistoryMamba = HistoryMamba(
#         patch_size=16,
#         embed_dim=576,
#         depth=32,
#         rms_norm=True,
#         residual_in_fp32=True,
#         fused_add_norm=True,
#         **kwargs
#     )
#     model.default_cfg = _cfg()
#     if pretrained:
#         print('load pretrained weights')
#         checkpoint = torch.load(model_name_or_path, map_location='cpu')
#         if 'model' in checkpoint:
#             state_dict = checkpoint['model']
#         else:
#             state_dict = checkpoint
#         videomamba.load_state_dict(model, state_dict, center=True)
#
#     return model

def build_history_mamba(
        config: PretrainedConfig,
        checkpoint_path: str = None,
) -> VisionMamba:

    model = VisionMamba(
        img_size=384,
        # patch_size=24,
        patch_size=32,
        in_chans=3,
        embed_dim=768,
        depth=4,
        dtype=config.model_dtype,
        # device=config.device,
    )


    if checkpoint_path is not None:
        loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(loaded_state_dict)

    return model




if __name__ == "__main__":
    model = build_history_mamba("/mnt/sdc/weiguanzhao/navila-llama3-8b-8f").cuda()

    x = torch.randn(1, 3, 8, 224, 224).cuda()

    with torch.no_grad():  # 推理时关闭梯度
        features = model.forward_features(x)  # ← 直接调用特征提取
        print("Feature shape:", features.shape)