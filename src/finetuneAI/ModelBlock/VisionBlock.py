from .ModelLoader import ModelLoader

from torch import Tensor

from transformers import AutoConfig
from transformers.modeling_outputs import ModelOutput

from PIL import Image
from typing import Any, Dict, Generic, TypeVar

T = TypeVar("T")

class _LazyTaskMapping(dict):

    _MAPPING_TASKS = {
        # ====== Computer Vision ======
        "classification"            : ("AutoModelForImageClassification", "AutoImageProcessor"),
        "detection"                 : ("AutoModelForObjectDetection", "AutoImageProcessor"),
        "basic-segmentation"        : ("AutoModelForImageSegmentation", "AutoImageProcessor"),
        "universal-segmentation"    : ("Mask2FormerForUniversalSegmentation", "AutoImageProcessor"),
        "semantic-segmentation"     : ("SegformerForSemanticSegmentation", "SegformerImageProcessor"),
    }

    def __getitem__(self, key):
        if key not in self._MAPPING_TASKS:
            raise KeyError(key)
        model, processor = self._MAPPING_TASKS[key]
        module = __import__("transformers", fromlist=[model, processor])
        return getattr(module, model), getattr(module, processor)

    def __contains__(self, key):
        return key in self._MAPPING_TASKS


    def __iter__(self):
        return iter(self._MAPPING_TASKS)

    def __len__(self):
        return len(self._MAPPING_TASKS)

    @property
    def keys(self):
        return self._MAPPING_TASKS.keys()

MODALITY_TO_TASK_MAPPING = _LazyTaskMapping()

class VisionBlock(ModelLoader[T]):

    _MAPPING_MODELS = {
        "YolosForObjectDetection"               : "detection",
        "DetrForObjectDetection"                : "detection",
        "AutoModelForImageClassification"       : "classification",
        "ViTForImageClassification"             : "classification",
        "Mask2FormerForUniversalSegmentation"   : "universal-segmentation",
        "SegformerForSemanticSegmentation"      : "semantic-segmentation",
        }


    def __init__(self):
        super().__init__()
        self.height, self.width = float, float

    def _detect_task(self, model_name:str):

        archs = getattr(self._config, "architectures", [])
        for arch in archs:
            if arch in self._MAPPING_MODELS:
                return self._MAPPING_MODELS[arch]

        mt = getattr(self._config, "model_type", "").lower()
        if "yolos" in mt or "detr" in mt:
            return "detection"
        if "vit" in mt or "resnet" in mt or "efficientnet" in mt:
            return "classification"

        raise ValueError(f"Unknown task for architecture {archs}")

    def load_model(self, model_name:str, **kwargs) -> None:
        self._config    = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.task       = self._detect_task(model_name)

        model_cls, processor_cls = MODALITY_TO_TASK_MAPPING[self.task]
        self._processor = processor_cls.from_pretrained(model_name, trust_remote_code=True)
        self._model     = model_cls.from_pretrained(model_name, trust_remote_code=True)

        return

    def preprocess(self, input:Any, **kwargs) -> Dict[str, Tensor]:

        if not isinstance(input, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image")

        self.height, self.width = input.height, input.width
        inputs = self._processor(images=input, return_tensors="pt")
        return inputs





