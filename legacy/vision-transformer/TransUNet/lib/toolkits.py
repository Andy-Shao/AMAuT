import os

import torch
import torch.nn as nn

def store_model_structure(model:nn.Module, inputs:torch.Tensor, output_path:str) -> None:
    from torch.utils.tensorboard.writer import SummaryWriter
    import shutil

    try:
        if os.path.exists(output_path): shutil.rmtree(output_path)
    except:
        pass

    writer = SummaryWriter(log_dir=output_path)
    writer.add_graph(model=model, input_to_model=inputs)
    writer.close()

def print_model(model:nn.Module, output_path:str) -> None:
    model_info = str(model)
    with open(file=output_path, mode='w') as f:
        f.write(model_info)