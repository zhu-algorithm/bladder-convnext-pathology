import timm


def build_model(cfg):
    return timm.create_model(cfg["model_name"], pretrained=bool(cfg.get("pretrained", True)),
                             num_classes=int(cfg.get("num_classes", 2)))

