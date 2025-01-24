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