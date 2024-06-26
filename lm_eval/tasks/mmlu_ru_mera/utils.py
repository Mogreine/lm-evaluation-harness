import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        output = doc["outputs"]
        return {"query": instruction.format(**inputs), "label": output}

    return dataset.map(_process_doc)


def process_docs_continuation(dataset: datasets.Dataset) -> datasets.Dataset:
    QUERY = """Ниже приведены вопросы с ответами на тему {subject}.
Вопрос: {text}
Ответ:"""
    option_keys = ["option_a", "option_b", "option_c", "option_d"]
    targets = ["A", "B", "C", "D"]

    def _process_doc(doc):
        inputs = doc["inputs"]
        return {
            "query": QUERY.format(subject=inputs["subject"], text=inputs["text"]),
            "choices": [inputs[it] for it in option_keys],
            "label": targets.index(doc["outputs"]),
        }

    return dataset.map(_process_doc)
