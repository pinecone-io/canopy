import abc
from typing import Optional, Type


class FactoryMixin:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            cls._SUPPORTED_CLASSES = {}
        if cls.__base__ is not abc.ABC:
            cls._SUPPORTED_CLASSES[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: dict, default_class = None):
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            raise ValueError("from_config() can only be called from a subclass.")

        if cls.__base__ is not abc.ABC:
            raise ValueError("from_config() can only be called from the base class.")

        if "type" in config:
            class_name = config["type"]
            if not class_name in cls._SUPPORTED_CLASSES:
                raise ValueError(
                    f"{class_name} is not supported. Allowed {cls.__name__}s are: "
                    f"{list(cls._SUPPORTED_CLASSES.keys())}"
                )

            class_type = cls._SUPPORTED_CLASSES[class_name]
        elif default_class:
            class_type = default_class
        else:
            raise ValueError(
                f"Error loading config for {cls.__name__}. Either specify 'type' in the"
                f" confi or provide a default_class."
            )
        return class_type(**config.get("params", {}))



class ConfigurableMixin:
    _DEFAULT_COMPONENTS = {}
    _UNALLOWED_CONFIG_KEYS = {}
    _MANDATORY_CONFIG_KEYS = {}

    @classmethod
    def from_config(cls,
                    config: dict,
                    overrides: dict):
        pass

def _load_encapsulated_component(config: dict,
                                 component_name: str,
                                 component_base: Type[FactoryMixin],
                                 default_class: Type[FactoryMixin],
                                 override_object):
    component_config = config.pop(component_name, None)
    if override_object and component_config:
        raise ValueError(f"Cannot specify both {component_name} and {override_object}")
    if override_object:
        return override_object

    if component_config:
        return component_base.from_config(component_config, default_class)

    return None
