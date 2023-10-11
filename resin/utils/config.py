import abc
from typing import Dict, Any


class FactoryMixin:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            cls._SUPPORTED_CLASSES = {}
        # if cls.__base__ is not abc.ABC:
        if FactoryMixin in cls.__bases__:
            cls.__FACTORY_BASE_CLASS__ = cls
        else:
            cls._SUPPORTED_CLASSES[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        if not (hasattr(cls, '_SUPPORTED_CLASSES') and FactoryMixin in cls.__bases__):
            raise ValueError("from_config() can only be called from the base class.")

        class_name = config.pop("type")
        if class_name is None:
            raise ValueError(
                f"{cls.__name__} load error: missing 'type' field in config."
            )
        if class_name not in cls._SUPPORTED_CLASSES:
            raise ValueError(
                f"{cls.__name__} load error: {class_name} is not supported. "
                f"Allowed values are: {list(cls._SUPPORTED_CLASSES.keys())}"
            )

        class_type = cls._SUPPORTED_CLASSES[class_name]

        if issubclass(class_type, ConfigurableMixin):
            return class_type.from_config(config)

        return class_type(**config.get("params", {}))

    @classmethod
    def list_supported_types(cls):
        return list(cls._SUPPORTED_CLASSES.keys())


class ConfigurableMixin(abc.ABC):
    _DEFAULT_COMPONENTS: Dict[str, type] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config: Dict[str, Any], **kwargs):
        loaded_components = {}
        for component_name, default_class in cls._DEFAULT_COMPONENTS.items():
            if component_name in kwargs:
                raise RuntimeError(
                    f"{cls.__name__} load error: Overriding {component_name} is not "
                    f"allowed. Please set it in the config file."
                )
            component_config = config.pop(component_name, {})
            assert issubclass(default_class, FactoryMixin)
            component_config['type'] = component_config.get(
                'type', default_class.__name__
            )

            # For classes implementing FactoryMixin, we need to call
            # `from_config()` on the base class
            assert hasattr(default_class, '__FACTORY_BASE_CLASS__')
            base_class = default_class.__FACTORY_BASE_CLASS__
            component = base_class.from_config(component_config)
            loaded_components[component_name] = component

        parameters = config.pop("params", {})
        params_in_kwargs = set(parameters.keys()) & set(kwargs.keys())
        if params_in_kwargs:
            raise RuntimeError(
                f"{cls.__name__} load error: can't set {params_in_kwargs} in both "
                f"config and constructor."
            )

        # The config should be empty at this point
        if config:
            allowed_keys = list(cls._DEFAULT_COMPONENTS.keys()) + ['params', 'type']
            raise ValueError(
                f"Unrecognized keys in {cls.__name__} config: {config.keys()}. "
                f"The allowed keys are: {allowed_keys}"
            )

        return cls(**loaded_components, **parameters, **kwargs)
