from os import path

import torch

try:
    from utils.dist import is_main_process
except Exception as e:
    print(e)


def save_model(fpn_model, optimizer, config, val_loss=0, step=0, model_only=False):
    if is_main_process():
        save_path = config.model_path

        if model_only:
            torch.save(
                {"fpn_model": fpn_model.state_dict()},
                path.join(save_path, f"model_val_loss_{val_loss:.4f}_step_{step}.pth"),
            )
        else:
            torch.save(
                {
                    "fpn_model": fpn_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path.join(save_path, f"model_val_loss_{val_loss:.4f}_step_{step}.pth"),
            )


def load_model(fpn_model, model_path, model_only=True, optimizer=None, cpu_mode=False):
    if cpu_mode:
        checkpoint = torch.load(model_path, map_location="cpu")
    else:
        checkpoint = torch.load(model_path)

    fpn_model.load_state_dict(checkpoint["fpn_model"])

    if not model_only:
        optimizer.load_state_dict(checkpoint["optimizer"])
