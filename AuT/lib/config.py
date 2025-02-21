from ml_collections import ConfigDict

def transformer_cfg(embed_size:int, cfg:ConfigDict) -> None:
    cfg.transform = ConfigDict()
    cfg.transform.layer_num = 12 if embed_size == 768 else 24
    cfg.transform.head_num = 12 if embed_size == 768 else 16
    cfg.transform.atten_drop_rate = .0
    cfg.transform.mlp_mid = 3072 if embed_size == 768 else 4096
    cfg.transform.mlp_dp_rt = .0
    
def classifier_cfg(class_num:int, cfg:ConfigDict) -> None:
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = class_num
    cfg.classifier.extend_size = 2048
    cfg.classifier.convergent_size = 256

def decoder_cfg(n_mels:int, embed_size:int, cfg:ConfigDict) -> None:
    decoder = ConfigDict()
    decoder.in_channels = [512, 128, 128]
    decoder.out_channels = [128, 128, n_mels]
    decoder.skip_channels = [512, 128, 0]
    decoder.hidden_size = embed_size
    cfg.decoder = decoder

def CT_base(class_num:int, n_mels:int) -> ConfigDict:
    cfg = ConfigDict()

    cfg.transform = ConfigDict()
    cfg.transform.layer_num = 12
    cfg.transform.head_num = 12
    cfg.transform.atten_drop_rate = .0
    cfg.transform.mlp_mid = 3072
    cfg.transform.mlp_dp_rt = .0

    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = class_num
    cfg.classifier.extend_size = 2048
    cfg.classifier.convergent_size = 256

    cfg.embedding = ConfigDict()
    cfg.embedding.channel_num = n_mels
    cfg.embedding.marsked_rate = .15
    cfg.embedding.embed_size = 768
    cfg.embedding.in_shape = [80, 104]
    cfg.embedding.arch = 'CT'
    cfg.embedding.num_layers = [6, 8]

    return cfg