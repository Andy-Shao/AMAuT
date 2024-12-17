import sys 

import torch 
import torch.nn as nn

def store_model_structure_to_txt(model: nn.Module, output_path: str) -> None:
    model_info = str(model)
    with open(output_path, 'w') as f:
        f.write(model_info)

def print_attributes(obj, atts:list[str], output_path:str) -> None:
    if atts is None or len(atts) == 0:
        lines = [f'{k}={v}\n' for k,v in obj.__dict__.items()]
    else:
        lines = []
        for att in atts:
            if hasattr(obj, att):
                v = getattr(obj, att)
                if isinstance(v, torch.Tensor):
                    lines.append(f'{att} -> {v.shape}\n')
                else:
                    lines.append(f'{att}={v}\n')
            else:
                lines.append(f'{att} does not exist\n')
    
    with open(output_path, 'w') as f:
        f.writelines(lines)