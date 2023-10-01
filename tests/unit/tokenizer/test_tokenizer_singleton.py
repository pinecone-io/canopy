import pytest
from resin.tokenizer import Tokenizer
from ..stubs.stub_tokenizer import StubTokenizer


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


def test_tokenizer_init_already_initialized(reset_tokenizer_singleton):
    Tokenizer.initialize(StubTokenizer)
    with pytest.raises(ValueError):
        Tokenizer.initialize(StubTokenizer)


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
