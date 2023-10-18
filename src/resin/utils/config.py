from typing import Dict, Any


class ConfigurableMixin:
    _DEFAULT_COMPONENTS: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_SUPPORTED_CLASSES'):
            cls._SUPPORTED_CLASSES = {}
        if ConfigurableMixin in cls.__bases__:
            cls.__FACTORY_BASE_CLASS__ = cls
        else:
            cls._SUPPORTED_CLASSES[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config: Dict[str, Any]):
        # Make a copy of the config, so we can modify it (e.g. pop fields) without
        # affecting the user's original config.
        config = config.copy()

        # These asserts should be true for all subclasses of ConfigurableMixin
        assert hasattr(cls, '_SUPPORTED_CLASSES')
        assert hasattr(cls, '__FACTORY_BASE_CLASS__')

        # If this is the base class, we expect a 'type' field in the config which will
        # tell us which derived class to load.
        if cls is cls.__FACTORY_BASE_CLASS__:
            derived_class = cls._get_derived_class(config)
            return derived_class.from_config(config)

        # If we got here, this is a derived class. We do not expect a 'type' field
        # in the config since we already know which class to load.
        if "type" in config:
            base_name = cls.__FACTORY_BASE_CLASS__.__name__
            raise ValueError(
                f"{cls.__name__} load error: 'type' field is not allowed in config. "
                f"if you wish to load another {base_name} subclass, use "
                f"`{base_name}.from_config(config)` instead."
            )

        # Load the class's subcomponents (dependencies) recursively
        loaded_components = cls._load_sub_components(config)

        parameters = config.pop("params", {})

        # The config should be empty at this point
        if config:
            allowed_keys = ['type', 'params'] + list(cls._DEFAULT_COMPONENTS.keys())
            raise ValueError(
                f"Unrecognized keys in {cls.__name__} config: {list(config.keys())}. "
                f"The allowed keys are: {allowed_keys}"
            )

        try:
            return cls(**loaded_components, **parameters)
        except TypeError as e:
            raise TypeError(
                f"{cls.__name__} load error: {e}. Please check the config."
            )

    @classmethod
    def _get_derived_class(cls, config):
        if "type" not in config:
            raise ValueError(
                f"{cls.__name__} load error: missing 'type' field in config."
            )
        derived_class_name = config.pop("type")
        if derived_class_name not in cls._SUPPORTED_CLASSES:
            raise ValueError(
                f"{cls.__name__} load error: {derived_class_name} is not supported."
                f" Supported types are: {list(cls._SUPPORTED_CLASSES.keys())}"
            )
        derived_class = cls._SUPPORTED_CLASSES[derived_class_name]
        return derived_class

    @classmethod
    def list_supported_types(cls):
        if cls is not cls.__FACTORY_BASE_CLASS__:
            raise RuntimeError(
                f"{cls.__name__} list_supported_types() should only be called on the "
                f"base class."
            )
        return list(cls._SUPPORTED_CLASSES.keys())

    @classmethod
    def _load_sub_components(cls, config):
        loaded_components = {}
        for component_name, default_class in cls._DEFAULT_COMPONENTS.items():
            component_config = config.pop(component_name, {})
            component_config['type'] = component_config.get(
                'type', default_class.__name__
            )

            # For classes implementing ConfigurableMixin, we need to call
            # `from_config()` on the base class
            assert hasattr(default_class, '__FACTORY_BASE_CLASS__')
            base_class = default_class.__FACTORY_BASE_CLASS__
            component = base_class.from_config(component_config)
            loaded_components[component_name] = component
        return loaded_components
