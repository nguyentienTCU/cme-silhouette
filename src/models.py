"""Existing SMP architectures and initialization settings."""

MODEL_NAMES = ('linknet', 'fpn', 'upernet', 'manet', 'deeplabv3plus')

def get_model_classes():
    import segmentation_models_pytorch as smp
    return {'linknet': smp.Linknet, 'fpn': smp.FPN, 'upernet': smp.UPerNet,
            'manet': smp.MAnet, 'deeplabv3plus': smp.DeepLabV3Plus}

def build_model(model_class):
    if isinstance(model_class, str):
        model_class = get_model_classes()[model_class]
    return model_class(encoder_name='resnet18', encoder_weights='imagenet',
                       in_channels=1, classes=1, activation=None)
