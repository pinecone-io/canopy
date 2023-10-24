import os

import pytest
import yaml

from canopy.chat_engine import ChatEngine
from canopy.context_engine import ContextEngine
from canopy.knowledge_base import KnowledgeBase

DEFAULT_COFIG_PATH = 'config/config.yaml'


@pytest.fixture(scope='module')
def temp_index_name():
    index_name_before = os.getenv("INDEX_NAME", None)

    os.environ["INDEX_NAME"] = "temp_index"
    yield "temp_index"

    if index_name_before is None:
        del os.environ["INDEX_NAME"]
    else:
        os.environ["INDEX_NAME"] = index_name_before


def test_default_config_matches_code_defaults(temp_index_name):
    with open(DEFAULT_COFIG_PATH) as f:
        default_config = yaml.safe_load(f)
    chat_engine_config = default_config['chat_engine']

    loaded_chat_engine = ChatEngine.from_config(chat_engine_config)
    default_kb = KnowledgeBase(index_name=temp_index_name)
    default_context_engine = ContextEngine(default_kb)
    default_chat_engine = ChatEngine(default_context_engine)

    def assert_identical_components(loaded_component, default_component):
        assert type(loaded_component) == type(default_component)  # noqa: E721
        if not loaded_component.__module__.startswith("canopy"):
            return

        for key, value in default_component.__dict__.items():
            assert hasattr(loaded_component, key), (
                f"Missing attribute {key} in {type(loaded_component)}"
            )
            if hasattr(value, '__dict__'):
                assert_identical_components(getattr(loaded_component, key), value)
            else:
                assert getattr(loaded_component, key) == value, (
                    f"Attribute {key} in {type(loaded_component)} is {value} in code "
                    f"but {getattr(loaded_component, key)} in config"
                )

    assert_identical_components(loaded_chat_engine, default_chat_engine)
