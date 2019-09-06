import abc


def get_concrete(baseclass):
    classes = baseclass.__subclasses__()
    for c in classes:
        classes += c.__subclasses__()
    final_classes = [c for c in classes if abc.ABC not in c.__bases__]
    return final_classes
