"""Model factories use standard serializable Keras layers only."""


def build_model(name, learning_rate=0.001, seed=42):
    if name == "simple_ann":
        from .simple_ann import build
    elif name == "cnn":
        from .cnn import build
    else:
        raise ValueError(f"Unknown model: {name}")
    return build(learning_rate, seed)
