from smb_biopan_utils import process_mm_info
from smb_biopan_utils.smb_vision.modeling_smb_vision import SMBVisionModel
from smb_biopan_utils.smb_vision.configuration_smb_vision import SMBVisionModelConfig
from transformers import AutoProcessor, AutoModel
import torch
from safetensors.torch import load_file as load_safetensors_file


if __name__ == "__main__":
    # initialize model
    # config = SMBVisionModelConfig(
    #     vision_config={
    #         "deepstack_visual_indexes": [8, 16, 24],
    #         "depth": 27,
    #         "hidden_act": "gelu_pytorch_tanh",
    #         "hidden_size": 1152,
    #         "in_channels": 1,
    #         "initializer_range": 0.02,
    #         "intermediate_size": 4304,
    #         "model_type": "smb_vision_encoder",
    #         "num_heads": 16,
    #         "num_position_embeddings": 2304,
    #         "out_hidden_size": 2048,
    #         "patch_size": 16,
    #         "spatial_merge_size": 2,
    #         "temporal_patch_size": 16,
    #     },
    #     predictor_config={
    #         "depth": 12,
    #         "in_hidden_size": 1152,
    #         "hidden_size": 512,
    #         "hidden_act": "gelu_pytorch_tanh",
    #         "intermediate_size": 1536,
    #         "num_heads": 16,
    #         "in_channels": 1,
    #         "initializer_range": 0.02,
    #     },
    #     masking_ratio=0.65,
    # )
    # model = SMBVisionModel(config)

    # load new state dict
    # new_state_dict = {}
    # weights = load_safetensors_file(
    #     "/workspace/checkpoints/qwen3-vl-30b-a3b-instruct/model-00013-of-00013.safetensors"
    # )
    # for key, value in weights.items():
    #     print(key, value.shape)
    #     new_key = key.replace("model.visual.", "encoder.")
    #     new_state_dict[new_key] = value
    # model.load_state_dict(new_state_dict, strict=False)
    # model.save_pretrained("/workspace/checkpoints/smb-vision-v0")

    # messages = [
    #     {
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "s3://smb-dev-us-east-2-data/datasets/idc2niix-ct/000009a8-6fb2-479f-bc17-d5bfe559703d/131937_1.2.840.113654.2.55.52839620203005001518305530405932392519_1.2.840.113654.2.55.174554873826805866673063405368371305947_Eq_1.nii.gz",
    #             },
    #             {
    #                 "type": "image",
    #                 "image": "s3://smb-dev-us-east-2-data/datasets/idc2niix-ct/000009a8-6fb2-479f-bc17-d5bfe559703d/131937_1.2.840.113654.2.55.52839620203005001518305530405932392519_1.2.840.113654.2.55.174554873826805866673063405368371305947_Eq_1.nii.gz",
    #             }
    #         ]
    #     }
    # ]
    # images, grid_thw = process_mm_info(messages)

    # initialize processor
    # processor = AutoProcessor.from_pretrained("standardmodelbio/SMB-30B-A3B-v1", trust_remote_code=True)

    # Preparation for inference
    # inputs = processor.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )

    # Process inputs
    # inputs = processor(text=inputs, images=images, return_tensors="pt").to("cuda")
    # for key, value in inputs.items():
    #     print(key, value.shape)

    # print(model)
    model = AutoModel.from_pretrained(
        "standardmodelbio/smb-vision-v0",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to("cuda")

    x = torch.randn(1600, 4096).to("cuda")
    grid_thw = torch.tensor([[4, 10, 10], [4, 10, 10], [4, 10, 10], [4, 10, 10]]).to("cuda")
    context_mask = torch.tensor([1, 1, 1, 1]).to("cuda")
    target_mask = torch.tensor([0, 1, 0, 1]).to("cuda")

    # # run vision encoder
    # # print(x[context_mask == 1])
    # outputs = model(x.to("cuda"), grid_thw=grid_thw.to("cuda"), context_mask=context_mask, target_mask=target_mask)
    # print(outputs)
    outputs = model.forward_features(x.to("cuda"), grid_thw=grid_thw.to("cuda"))
    print(outputs)
    # print(outputs.hidden_states.shape)
    # print(outputs.predicted_hidden_states.shape)
    # print(outputs.mim_loss)
    # print(outputs.jepa_loss)
    # print(outputs.loss)
