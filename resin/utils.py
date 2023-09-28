import abc
from typing import Optional, Type, Union
import logging

class FactoryMixin:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            cls._SUPPORTED_CLASSES = {}
        if cls.__base__ is not abc.ABC:
            cls._SUPPORTED_CLASSES[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: dict, default_class = None):
        if not (hasattr(cls, '_SUPPORTED_CLASSES') and cls.__base__ is abc.ABC):
            raise ValueError("from_config() can only be called from the base class.")

        if "type" in config:
            class_name = config["type"]
            if class_name not in cls._SUPPORTED_CLASSES:
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
                f" config or provide a default_class."
            )
        return class_type(**config.get("params", {}))



class ConfigurableMixin(abc.ABC):
    _DEFAULT_COMPONENTS = {}
    # _UNALLOWED_CONFIG_KxEYS = {}
    _MANDATORY_CONFIG_KEYS = {}

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict):
        pass

    def _set_component(self,
                       base_class: type,
                       component_name: str,
                       component):
        class_name = self.__class__.__name__
        logger = logging.getLogger(class_name)
        if component:
            if not isinstance(component, base_class):
                raise ValueError(
                    f"{class_name}: {component} must be an instance of {base_class}"
                )
            return component
        else:
            default_class = self._DEFAULT_COMPONENTS[component_name]
            logger.info(f"{class_name}: Created using default {component_name}: "
                        f"{default_class}")
            return default_class()

    @classmethod
    def _from_config(cls,
                     config: dict,
                     **kwargs):
        missing_keys = set(cls._MANDATORY_CONFIG_KEYS) - set(config.keys())
        if missing_keys:
            raise ValueError(f"")

        loaded_components = {}
        for component_name in cls._DEFAULT_COMPONENTS:
            override = kwargs.get(component_name, None)
            component_config = config.pop(component_name, None)
            if component_config and override:
                raise ValueError(f"Cannot both provide {component_name} override and config."
                                 f" If you want to use your own {component_name} - remove"
                                 f" the {component_name} key from the config")
            if override:
                 component = override
            elif component_config:
                default_class = cls._DEFAULT_COMPONENTS[component_name]
                if issubclass(default_class, FactoryMixin):
                    component = default_class.__base__.from_config(component_config,
                                                                   default_class)
                elif issubclass(default_class, ConfigurableMixin):
                    component = default_class.from_config(component_config)
            else:
                pass
            loaded_components[component_name] = component

        return cls(**kwargs, **config)


        # unallowed_keys = set(kwargs.keys()).intersection(set(config.keys()))
        # if unallowed_keys:
        #     raise ValueError(f"These keys are not allowed in {cls._NAME} config")




def _load_component_from_config(config: dict,
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
