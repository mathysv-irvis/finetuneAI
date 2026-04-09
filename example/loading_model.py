from finetuneAI import VisionBlock
import requests
from PIL import Image

model_name = "hustvl/yolos-tiny"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

vision_block = VisionBlock()
vision_block.load_model(model_name)

preprocessed_im = vision_block.preprocess(image)
print(preprocessed_im)
