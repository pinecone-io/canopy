import abc
from typing import Dict, Any
import logging


class FactoryMixin:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            cls._SUPPORTED_CLASSES = {}
        if cls.__base__ is not abc.ABC:
            cls._SUPPORTED_CLASSES[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: dict, default_class=None):
        if not (hasattr(cls, '_SUPPORTED_CLASSES') and cls.__base__ is abc.ABC):
            raise ValueError("from_config() can only be called from the base class.")

        if "type" in config:
            class_name = config.pop("type")
            if class_name not in cls._SUPPORTED_CLASSES:
                raise ValueError(
                    f"{cls.__name__} load error: {class_name} is not supported. "
                    f"Allowed values are: {list(cls._SUPPORTED_CLASSES.keys())}"
                )

            class_type = cls._SUPPORTED_CLASSES[class_name]
        elif default_class:
            class_type = default_class
        else:
            raise ValueError(
                f"Error loading config for {cls.__name__}. Either specify 'type' in the"
                f" config or provide a default_class."
            )

        if issubclass(class_type, ConfigurableMixin):
            return class_type.from_config(config)

        return class_type(**config.get("params", {}))

    @classmethod
    def list_supported_types(cls):
        return list(cls._SUPPORTED_CLASSES.keys())


class ConfigurableMixin(abc.ABC):
    _DEFAULT_COMPONENTS: Dict[str, type] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):  # type: ignore[override]
        return cls._from_config(config)

    def _set_component(self,
                       base_class: type,
                       component_name: str,
                       component):
        class_name = self.__class__.__name__
        logger = logging.getLogger(class_name)
        if component:
            if not isinstance(component, base_class):
                raise TypeError(
                    f"{class_name}: {component_name} must be an instance "
                    f"of {base_class.__name__}"
                )
            return component
        else:
            default_class = self._DEFAULT_COMPONENTS[component_name]
            logger.info(f"{class_name}: Created using default {component_name}: "
                        f"{default_class}")
            return default_class()

    @classmethod
    def _from_config(cls, config: Dict[str, Any]):
        loaded_components = {}
        for component_name in cls._DEFAULT_COMPONENTS:
            component_config = config.pop(component_name, None)
            if component_config:
                default_class = cls._DEFAULT_COMPONENTS[component_name]
                if issubclass(default_class, FactoryMixin):
                    base_class = default_class
                    while base_class.__base__ is not abc.ABC:
                        base_class = base_class.__base__
                    component = base_class.from_config(component_config, default_class)
                else:
                    raise ValueError(
                        f"{cls.__name__} load error: cannot load {component_name} from "
                        f"config"
                    )
                loaded_components[component_name] = component

        parameters = config.pop("params", {})

        # The config should be empty at this point
        if config:
            allowed_keys = list(cls._DEFAULT_COMPONENTS.keys()) + ['params', 'type']
            raise ValueError(
                f"Unrecognized keys in {cls.__name__} config: {config.keys()}. "
                f"The allowed keys are: {allowed_keys}"
            )

        return cls(**loaded_components, **parameters)
