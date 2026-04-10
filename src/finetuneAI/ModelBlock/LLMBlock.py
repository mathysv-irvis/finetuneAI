from ModelLoader import ModelLoader, _LazyTaskMapping

_MAPPING_TASKS = {
    # ====== LLM ======
    "llm"                       : ("AutoModelForCausalLM", "AutoTokenizer")
}

MODALITY_TO_TASK_MAPPING = _LazyTaskMapping()

