import ml_collections

def get_l8_config():
    """Returns the AuT-L/8 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_l8_config():
    """Returns the Resnet50 + AuT-L/8 configuration. customized """
    config = get_l8_config()
    config.patches.grid = (8, 8)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (7, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_b8_config():
    """Returns the AuT-B/8 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b8_config():
    """Returns the Resnet50 + AuT-B/8 configuration."""
    config = get_b8_config()
    config.patches.grid = (8, 8)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (7, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config