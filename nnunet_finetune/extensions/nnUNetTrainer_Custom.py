import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

import os

class nnUNetTrainer_Custom(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 150
        self.model_checkpoints_dir = os.path.join(
            self.output_folder, "model_checkpoints"
        )
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)

    def on_epoch_end(self):
        if (self.current_epoch + 1) % 10 == 0:
            self.save_checkpoint(
                os.path.join(self.model_checkpoints_dir, f"checkpoint_epoch{self.current_epoch + 1}.pth")
            )

            # TODO: maybe evaluation here? we'll see..

        super().on_epoch_end()