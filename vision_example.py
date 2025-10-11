from smb_biopan_utils import process_mm_info
from smb_biopan_utils.smb_vision.modeling_smb_vision import SMBVisionModel
from smb_biopan_utils.smb_vision.configuration_smb_vision import SMBVisionModelConfig
from transformers import AutoProcessor
import torch


if __name__ == "__main__":
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "/workspace/data/PE1677ce5.nii.gz"
    #             },
    #             {
    #                 "type": "image",
    #                 "image": "/workspace/data/PE1677ce5.nii.gz"
    #             },
    #             {
    #                 "type": "text",
    #                 "text": "What the heck is going on here?"
    #             }
    #         ]
    #     }
    # ]
    # images, grid_thw = process_mm_info(messages)
    # print(images.shape)
    # print(grid_thw.shape)

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

    # initialize model
    config = SMBVisionModelConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 384,
            "num_heads": 8,
            "in_channels": 1,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 4,
            "out_hidden_size": 128,
            "num_position_embeddings": 4400,
            "deepstack_visual_indexes": [1],
            "initializer_range": 0.02,
        },
        predictor_config={
            "depth": 2,
            "in_hidden_size": 128,
            "hidden_size": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 384,
            "num_heads": 8,
            "in_channels": 1,
            "initializer_range": 0.02,
        },
        masking_ratio=0.1,
    )
    model = SMBVisionModel(config).to("cuda")

    x = torch.randn(1600, 64).to("cuda")
    grid_thw = torch.tensor([[4, 10, 10], [4, 10, 10], [4, 10, 10], [4, 10, 10]]).to("cuda")
    context_mask = torch.tensor([1, 1, 1, 1]).to("cuda")
    target_mask = torch.tensor([0, 0, 1, 1]).to("cuda")

    # run vision encoder
    # print(x[context_mask == 1])
    outputs = model(x, grid_thw=grid_thw, context_mask=context_mask, target_mask=target_mask)
    print(outputs.hidden_states.shape)
    print(outputs.predicted_hidden_states.shape)
    print(outputs.mim_loss)
    print(outputs.jepa_loss)
    print(outputs.loss)
