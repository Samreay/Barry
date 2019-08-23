import pytest
import importlib
import inspect


def test_all_datasets_load():
    print(importlib.import_module("barry.datasets").__all__)

    from barry.datasets import Dataset
    Dataset.__subclasses__()

    inspect.isclass()

    Dataset.__subclasses__()[1].__subclasses__()[0].__subclasses__()