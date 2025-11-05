import yaml

from models.Transformers import ModelConfig, TinyLLM


def build_model(config_path: str) -> TinyLLM:
    """
    Build based on yaml
    """
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    model_cfg = ModelConfig(**yaml_data)

    model = TinyLLM(model_cfg)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")

    return model
