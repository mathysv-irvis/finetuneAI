from torch import float16, device, cuda, Tensor
from typing import Dict, Any, Generic, TypeVar, Optional
from abc import ABC, abstractmethod
from datasets import Dataset

T = TypeVar("T")

class _LazyTaskMapping(dict):

    _MAPPING_TASKS = {
        # ====== Computer Vision ======
        "classification"            : ("AutoModelForImageClassification", "AutoImageProcessor"),
        "detection"                 : ("AutoModelForObjectDetection", "AutoImageProcessor"),
        "basic-segmentation"        : ("AutoModelForImageSegmentation", "AutoImageProcessor"),
        "universal-segmentation"    : ("Mask2FormerForUniversalSegmentation", "AutoImageProcessor"),
        "semantic-segmentation"     : ("SegformerForSemanticSegmentation", "SegformerImageProcessor"),
        # ====== LLM ======
        "llm"                       : ("AutoModelForCausalLM", "AutoTokenizer")
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

class AIBlock(ABC, Generic[T]):

    MODALITY_TO_TASK_MAPPING = _LazyTaskMapping()

    def __init__(self):
        self._device = device("cuda" if cuda.is_available() else "cpu")
        self._model : Optional[T]    = None
        self._tokenizer : Optional   = None
        self._processor : Optional     = None

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @abstractmethod
    def preprocess(self, input:Any, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *args, **kwargs) -> None:
        raise NotImplementedError


