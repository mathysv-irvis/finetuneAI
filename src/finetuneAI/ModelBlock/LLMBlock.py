
class _LazyTaskMapping(dict):

    _MAPPING_TASKS = {
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


MODALITY_TO_TASK_MAPPING = _LazyTaskMapping()

