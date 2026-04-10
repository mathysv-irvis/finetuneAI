from torch import float16, device, cuda, Tensor
from typing import Dict, Any, Generic, TypeVar, Optional
from abc import ABC, abstractmethod
from datasets import Dataset

T = TypeVar("T")

class ModelLoader(ABC, Generic[T]):

    _ALIASES = {
        "LLMBlock"      : "tokenizer",
        "VisionBlock"   : "processor"
    }

    def __init__(self):
        self._device = device("cuda" if cuda.is_available() else "cpu")
        self._model : Optional[T]   = None
        self._model_name : str      = None
        self._tokenizer  : Any      = None
        self._processor  : Any      = None

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def tokenizer(self):
        if self.__getattr__('_tokenizer') == None :
            raise AttributeError("LLM model not loaded yet, has no tokenizer yet !")
        return self._tokenizer

    @property
    def processor(self):
        if self.__getattr__('_processor') == None:
            raise AttributeError("Vision model not loaded yet, has no processor yet !")
        return self._processor

    def __getattr__(self, name):
        cls_name = self.__class__.__name__
        if name in self._ALIASES.values():
            attr_str = self._ALIASES[cls_name]
            if attr_str != str(name):
                raise AttributeError(f"{cls_name} has no attribute {str(name)}. Use {attr_str} instead.", name=None)
            
            if name == "tokenizer":
                attr = self._tokenizer
            elif name == "processor":
                attr = self._processor
            if attr == None:
                raise AttributeError(f"{cls_name} not loaded yet, has no {attr_str} yet !", name=None)
            else :
                return attr

        return self.__getattribute__(name)

    @abstractmethod
    def preprocess(self, input:Any, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *args, **kwargs) -> None:
        raise NotImplementedError


