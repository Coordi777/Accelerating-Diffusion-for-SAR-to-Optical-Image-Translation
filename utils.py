import ast
from safetensors import safe_open
import torch
from dataclasses import dataclass
from typing import Optional, Union, List

def update_args_from_yaml(group, args, parser):
    for key, value in group.items():
        if isinstance(value, dict):
            update_args_from_yaml(value, args, parser)
        else:
            if value == 'None' or value == 'null':
                value = None
            else:
                arg_type = next((action.type for action in parser._actions if action.dest == key), str)
                
                if arg_type is ast.literal_eval:
                    pass
                elif arg_type is not None and not isinstance(value, arg_type):
                    try:
                        value = arg_type(value)
                    except ValueError as e:
                        raise ValueError(f"Cannot convert {key} to {arg_type}: {e}")

            setattr(args, key, value)


def safe_load(model_path):
    assert "safetensors" in model_path
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k) 
    return state_dict


