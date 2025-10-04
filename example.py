from smb_biopan_utils import process_mm_info
from smb_biopan_utils.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from smb_biopan_utils.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
from transformers import AutoProcessor
import torch


if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/root/data/PE1677ce5.nii.gz"
                },
                {
                    "type": "text",
                    "text": "What the heck is going on here?"
                }
            ]
        }
    ]
    images = process_mm_info(messages)
    images = torch.stack(images)
    print(images.shape)

    # initialize model
    config = Qwen3VLMoeConfig(
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "decoder_sparse_step": 1,
            "dtype": "float32",
            "eos_token_id": 151645,
            "head_dim": 20,
            "hidden_act": "silu",
            "hidden_size": 120,
            "initializer_range": 0.02,
            "intermediate_size": 360,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_vl_moe_text",
            "moe_intermediate_size": 120,
            "norm_topk_prob": True,
            "num_attention_heads": 16,
            "num_experts": 12,
            "num_experts_per_tok": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [
                2,
                2,
                2
            ],
            "rope_type": "default"
            },
            "rope_theta": 5000000,
            "use_cache": True,
            "vocab_size": 151936
        },
        vision_config={
            "deepstack_visual_indexes": [
                1,
            ],
            "depth": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 120,
            "in_channels": 1,
            "initializer_range": 0.02,
            "intermediate_size": 360,
            "model_type": "qwen3_vl_moe",
            "num_heads": 6,
            "num_position_embeddings": 2304,
            "out_hidden_size": 120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 16
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
    )
    model = Qwen3VLMoeForConditionalGeneration(config).to("cuda")

    # initialize processor
    processor = AutoProcessor.from_pretrained("standardmodelbio/SMB-RAD-v1", trust_remote_code=True)

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process inputs
    inputs = processor(text=inputs, images=images, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        print(key, value.shape)

    ## decode input ids
    # print(inputs["input_ids"])
    # print(processor.batch_decode(inputs["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False))

    ## generate
    outputs = model(**inputs)
    print(outputs)