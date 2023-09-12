def get_class_from_config(config: dict,
                          classes_dict: dict,
                          default_class: type,
                          name: str) -> object:
    if "type" in config:
        class_name = config["type"]
        if class_name in classes_dict:
            class_type = classes_dict[class_name]
        else:
            raise ValueError(
                f"Error reading {name} config: {class_name} not found in {classes_dict}"
                f". Allowed values are: {classes_dict.keys()}"
            )
    else:
        class_type = default_class

    params = config.get("params", {})
    return class_type(**params)