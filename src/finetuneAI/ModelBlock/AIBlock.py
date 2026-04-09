from torch import float16, device, cuda, Tensor
from typing import Dict, Any, Generic, TypeVar, Optional
from abc import ABC, abstractmethod
from datasets import Dataset

T = TypeVar("T")

class AIBlock(ABC, Generic[T]):

    def __init__(self):
        self._device = device("cuda" if cuda.is_available() else "cpu")
        self._model : Optional[T]    = None
        self._tokenizer : Optional   = None
        self._processor : Optional   = None

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


