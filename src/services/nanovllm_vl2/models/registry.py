MODEL_REGISTRY = {}

def register_model(model_name: str):
    def decorator(cls):
        MODEL_REGISTRY[model_name] = cls
        return cls
    return decorator

def get_model_class(hf_config):
    model_type = hf_config.model_type
    return MODEL_REGISTRY.get(model_type)