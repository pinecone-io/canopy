def type_from_str(type_str: str, type_dict: dict, name: str) -> type:
    if type_str not in type_dict:
        raise ValueError(f"Unknown {name}: {type_str}. Allowed values are {list(type_dict.keys())}")
    return type_dict[type_str]
