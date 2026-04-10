from finetuneAI import FinetuneAI, VisionBlock
from datasets import load_dataset

model_name   = "hustvl/yolos-tiny"
dataset_name = "duality-robotics/YOLOv8-Object-Detection-02-Dataset"

dataset = load_dataset(dataset_name, name="default")

print(dataset["train"][:10])
finetune_block = FinetuneAI(model_name)

finetune_block.train(dataset["train"], epoch=1)
finetune_block.merge()
