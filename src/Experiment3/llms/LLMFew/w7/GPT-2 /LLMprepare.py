import json
import torch
from transformers import AutoConfig, AutoModel, Phi3Model, Phi3Config
from peft import LoraConfig, get_peft_model

def LLMprepare(configs):
    model_path = "gpt2" 
    d_model = 768  

    llm_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Aplicar LoRA si está activado
    if configs.lora:
        lora_config = {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["c_attn"],
            "lora_dropout": 0.1,
            "bias": "none"
        }
        llm_model = get_peft_model(llm_model, LoraConfig(**lora_config))

        for name, param in llm_model.named_parameters():
            param.requires_grad = ('lora' in name)
    else:
        for name, param in llm_model.named_parameters():
            param.requires_grad = False

    return llm_model, d_model

