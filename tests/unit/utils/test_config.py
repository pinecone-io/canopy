# noqa: F405

import pytest

from ._stub_classes import (BaseStubChunker, StubChunker, StubKB,
                            BaseStubContextEngine, StubOtherChunker, StubContextBuilder,
                            StubContextEngine, BaseStubKB)


def test_list_supported_classes():
    supported_chunkers = BaseStubChunker.list_supported_types()
    assert len(supported_chunkers) == 2
    assert supported_chunkers == ['StubChunker', 'StubOtherChunker']

    supported_kbs = BaseStubKB.list_supported_types()
    assert len(supported_kbs) == 1
    assert supported_kbs == ['StubKB']


def test_list_supported_derived_class():
    with pytest.raises(RuntimeError) as e:
        StubChunker.list_supported_types()
    assert 'StubChunker' in str(e.value)
    assert 'base class' in str(e.value)


def _assert_chunker(chunker,
                    expected_type=StubChunker,
                    expected_chunk_size=StubChunker.DEFAULT_CHUNK_SIZE,
                    expected_splitter=StubChunker.DEFAULT_SPLITTER):
    assert isinstance(chunker, expected_type)
    assert chunker.chunk_size == expected_chunk_size
    assert chunker.splitter == expected_splitter


def test_from_config():
    config = {'type': 'StubChunker', 'params': {'chunk_size': 200, 'splitter': ','}}
    chunker = BaseStubChunker.from_config(config)
    _assert_chunker(chunker, expected_chunk_size=200, expected_splitter=',')


def test_from_config_partial_params():
    config = {'type': 'StubChunker', 'params': {'chunk_size': 300}}
    chunker = BaseStubChunker.from_config(config)
    _assert_chunker(chunker, expected_chunk_size=300)

    config['type'] = 'StubOtherChunker'
    other_chunker = BaseStubChunker.from_config(config)
    _assert_chunker(other_chunker,
                    expected_type=StubOtherChunker,
                    expected_chunk_size=300)


def test_from_config_default_params():
    config = {'type': 'StubChunker', 'params': {}}
    chunker = BaseStubChunker.from_config(config)
    _assert_chunker(chunker)


def test_from_config_no_params():
    config = {'type': 'StubChunker'}
    chunker = BaseStubChunker.from_config(config)
    _assert_chunker(chunker)


def test_from_config_no_type():
    config = {'params': {'chunk_size': 200, 'splitter': ','}}
    with pytest.raises(ValueError) as e:
        BaseStubChunker.from_config(config)
    assert 'type' in str(e.value)
    assert 'BaseStubChunker' in str(e.value)


def test_from_config_non_existing_param():
    config = {
        'type': 'StubChunker',
        'params': {'chunk_size': 200, 'non_existing_param': 'some_value'}
    }
    with pytest.raises(TypeError) as e:
        BaseStubChunker.from_config(config)
    assert 'non_existing_param' in str(e.value)
    assert 'StubChunker' in str(e.value)


def test_from_config_non_existing_type():
    config = {'type': 'NonExistingChunker', 'params': {'chunk_size': 200}}
    with pytest.raises(ValueError) as e:
        BaseStubChunker.from_config(config)
    assert 'NonExistingChunker' in str(e.value)
    assert 'BaseStubChunker' in str(e.value)
    supported_types = BaseStubChunker.list_supported_types()
    assert f"Supported types are: {supported_types}" in str(e.value)


def test_from_config_unsupported_key():
    config = {'type': 'StubChunker',
              'params': {'chunk_size': 200},
              'unsupported_key': 'some_value'}
    with pytest.raises(ValueError) as e:
        BaseStubChunker.from_config(config)
    assert 'unsupported_key' in str(e.value)
    assert 'StubChunker' in str(e.value)


def test_from_config_misspelled_key():
    config = {
        'type': 'StubChunker',
        'parms': {'chunk_size': 200},
    }
    with pytest.raises(ValueError) as e:
        BaseStubChunker.from_config(config)
    assert 'parms' in str(e.value)
    assert 'StubChunker' in str(e.value)
    assert "['type', 'params']" in str(e.value)


def test_from_config_derived_class():
    config = {'params': {'chunk_size': 200, 'splitter': ','}}
    chunker = StubChunker.from_config(config)
    _assert_chunker(chunker, expected_chunk_size=200, expected_splitter=',')


def test_from_config_derived_class_no_params():
    config = {}
    chunker = StubChunker.from_config(config)
    _assert_chunker(chunker)


def test_from_config_derived_class_partial_params():
    config = {'params': {'chunk_size': 300}}
    chunker = StubChunker.from_config(config)
    _assert_chunker(chunker, expected_chunk_size=300)


def test_from_config_derived_class_with_type():
    config = {'type': 'StubChunker', 'params': {'chunk_size': 200, 'splitter': ','}}
    with pytest.raises(ValueError) as e:
        StubChunker.from_config(config)
    assert 'type' in str(e.value)
    assert 'BaseStubChunker' in str(e.value)
    assert 'BaseStubChunker.from_config(' in str(e.value)


def test_init_with_default_params():
    chunker = StubChunker()
    _assert_chunker(chunker)


def _assert_kb(kb,
               expected_type=StubKB,
               expected_top_k=StubKB.DEFAULT_TOP_K,
               expected_chunker=None):
    assert isinstance(kb, expected_type)
    assert kb.top_k == expected_top_k
    expected_chunker = expected_chunker or {}
    _assert_chunker(kb.chunker, **expected_chunker)


_non_default_chunker_config = {
            'type': 'StubOtherChunker',
            'params': {'chunk_size': 200, 'some_param': ','}
        }

_non_default_kb_config = {
        'type': 'StubKB',
        'params': {'top_k': 10},
        'chunker': _non_default_chunker_config,
    }


def test_from_config_with_components():
    config = _non_default_kb_config

    kb = BaseStubKB.from_config(config)
    _assert_kb(kb,
               expected_top_k=10,
               expected_chunker={
                   'expected_type': StubOtherChunker,
                   'expected_chunk_size': 200,
                   'expected_splitter': ','
               })


def test_from_config_with_components_default():
    config = {'type': 'StubKB'}
    kb = BaseStubKB.from_config(config)
    _assert_kb(kb)


def test_from_config_with_components_derived_default():
    config = {}
    kb = StubKB.from_config(config)
    _assert_kb(kb)


def test_from_config_with_components_derived_partial():
    config = {'params': {'top_k': 20}}
    kb = StubKB.from_config(config)
    _assert_kb(kb, expected_top_k=20)


def test_from_config_with_components_unsupported_keys():
    config = {
        'type': 'StubKB',
        'params': {'top_k': 10},
        'chunker': {
            'type': 'StubChunker',
            'params': {'chunk_size': 200, 'splitter': ','}
        },
        'unsupported_key': 'some_value'
    }
    with pytest.raises(ValueError) as e:
        BaseStubKB.from_config(config)
    assert 'unsupported_key' in str(e.value)
    assert 'StubKB' in str(e.value)
    assert "['type', 'params', 'chunker']" in str(e.value)


def test_from_config_with_components_unsupported_component_type():
    config = {
        'type': 'StubKB',
        'params': {'top_k': 10},
        'chunker': {
            'type': 'NonExistingChunker',
        },
    }
    with pytest.raises(ValueError) as e:
        BaseStubKB.from_config(config)
    assert 'NonExistingChunker' in str(e.value)
    assert 'BaseStubChunker' in str(e.value)
    assert "['StubChunker', 'StubOtherChunker']" in str(e.value)


def test_init_with_components_default():
    kb = StubKB()
    _assert_kb(kb)


def _assert_context_builder(context_builder,
                            expected_type=StubContextBuilder,
                            max_context_length=StubContextBuilder.DEFAULT_MAX_CONTEXT_LENGTH, # noqa E501
                            ):
    assert isinstance(context_builder, expected_type)
    assert context_builder.max_context_length == max_context_length


def _assert_context_engine(context_engine,
                           expected_type=StubContextEngine,
                           expected_kb=None,
                           expected_context_builder=None,
                           expected_filter=None):
    assert isinstance(context_engine, expected_type)
    expected_kb = expected_kb or {}
    _assert_kb(context_engine.knowledge_base, **expected_kb)
    expected_context_builder = expected_context_builder or {}
    _assert_context_builder(context_engine.context_builder, **expected_context_builder)
    assert context_engine.filter == expected_filter


def test_from_config_complex():
    config = {
        'type': 'StubContextEngine',
        'params': {
            'filter': 'some_filter',
        },
        'knowledge_base': _non_default_kb_config,
        'context_builder': {
            'type': 'StubContextBuilder',
            'params': {'max_context_length': 100},
        },
    }

    context_engine = BaseStubContextEngine.from_config(config)
    _assert_context_engine(context_engine,
                           expected_kb={
                               'expected_top_k': 10,
                               'expected_chunker': {
                                   'expected_type': StubOtherChunker,
                                   'expected_chunk_size': 200,
                                   'expected_splitter': ','
                               },
                           },
                           expected_context_builder={
                               'expected_type': StubContextBuilder,
                               'max_context_length': 100,
                           },
                           expected_filter='some_filter')


def test_from_config_complex_default():
    config = {'type': 'StubContextEngine'}
    context_engine = BaseStubContextEngine.from_config(config)
    _assert_context_engine(context_engine)


def test_from_config_complex_derived_default():
    config = {}
    context_engine = StubContextEngine.from_config(config)
    _assert_context_engine(context_engine)


def test_from_config_complex_derived_partial():
    config = {'params': {'filter': 'some_filter'}}
    context_engine = StubContextEngine.from_config(config)
    _assert_context_engine(context_engine, expected_filter='some_filter')


def test_from_config_complex_unsupported_keys():
    config = {
        'type': 'StubContextEngine',
        'params': {'filter': 'some_filter'},
        'knowledge_base': _non_default_kb_config,
        'unsupported_key': 'some_value'
    }
    with pytest.raises(ValueError) as e:
        BaseStubContextEngine.from_config(config)
    assert 'unsupported_key' in str(e.value)
    assert 'StubContextEngine' in str(e.value)
    assert "['type', 'params', 'knowledge_base', 'context_builder']" in str(e.value)


def test_init_complex_default():
    kb = StubKB()
    context_engine = StubContextEngine(kb)
    _assert_context_engine(context_engine)


def test_init_complex_missing_mandatory_dependency():
    with pytest.raises(TypeError) as e:
        StubContextEngine()
    assert 'knowledge_base' in str(e.value)
