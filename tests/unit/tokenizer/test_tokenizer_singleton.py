import pytest
from canopy.tokenizer import Tokenizer
from ..stubs.stub_tokenizer import StubTokenizer


class StubChildTokenizer(StubTokenizer):
    pass


@pytest.fixture
def reset_tokenizer_singleton():
    before = Tokenizer._tokenizer_instance.__class__
    Tokenizer.clear()
    yield
    Tokenizer.clear()
    Tokenizer.initialize(tokenizer_class=before)


def test_tokenizer_init(reset_tokenizer_singleton):
    Tokenizer.initialize(StubTokenizer)
    assert isinstance(Tokenizer._tokenizer_instance, StubTokenizer)
    assert Tokenizer._initialized is True


def test_tokenizer_init_already_initialized_same_class(reset_tokenizer_singleton):
    Tokenizer.initialize(StubTokenizer, message_overhead=10)
    tokenizer = Tokenizer()
    assert isinstance(Tokenizer._tokenizer_instance, StubTokenizer)
    assert Tokenizer._initialized is True
    assert Tokenizer._tokenizer_instance._message_overhead == 10
    assert tokenizer._tokenizer_instance._message_overhead == 10


def test_tokenizer_init_already_initialized_different_class(reset_tokenizer_singleton):
    Tokenizer.initialize(StubChildTokenizer, message_overhead=10)
    tokenizer = Tokenizer()
    assert isinstance(Tokenizer._tokenizer_instance, StubChildTokenizer)
    assert Tokenizer._initialized is True
    assert isinstance(tokenizer._tokenizer_instance, StubChildTokenizer)


def test_tokenizer_init_invalid_same_class(reset_tokenizer_singleton):
    with pytest.raises(ValueError):
        Tokenizer.initialize(Tokenizer)


def test_tokenizer_init_invalid_tokenizer_class(reset_tokenizer_singleton):
    class InvalidTokenizer:
        pass
    with pytest.raises(ValueError):
        Tokenizer.initialize(InvalidTokenizer)


def test_tokenizer_uniqueness(reset_tokenizer_singleton):
    Tokenizer.initialize(StubTokenizer)
    tokenizer = Tokenizer()
    assert tokenizer is Tokenizer()
    another_tokenizer = Tokenizer()
    assert tokenizer is another_tokenizer
