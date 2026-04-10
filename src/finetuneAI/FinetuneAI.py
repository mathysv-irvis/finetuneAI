from transformers.modeling_outputs import ModelOutput
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from .ModelBlock import VisionBlock
import torch



class FinetuneAI:

    def __init__(self, model_name):
        self._model_name   = model_name
        self._model_loader = VisionBlock()
        self._model_loader.load_model(self._model_name)
        self._model_base   = self._model_loader.model
        self._trainer      = None
            
    def train(self, dataset, epoch=1, batch_size=4, lr=5e-5):
        processor = self._model_loader.processor
        model = self._model_loader.model

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epoch,
            learning_rate=lr,
            logging_steps=10,
            save_steps=50,
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        self.trainer.train()

    def merge(self, save_path="./finetuned_model"):
        if self.trainer is None:
            raise ValueError("Model not trained yet")

        self.trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)

        return {
            "model": self.model,
            "processor": self.processor,
            "path": save_path
        }
