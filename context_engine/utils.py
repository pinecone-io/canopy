from typing import Dict

TypeDict = Dict[str, type]


def initialize_from_config(config: dict,
                           classes_dict: TypeDict,
                           name: str):
    class_name = config.get("type", "default")
    if class_name in classes_dict:
        class_type = classes_dict[class_name]
    else:
        raise ValueError(
            f"Error reading {name} config: {class_name} not found in {classes_dict}"
            f". Allowed values are: {classes_dict.keys()}"
        )

    params = config.get("params", {})
    return class_type(**params)
