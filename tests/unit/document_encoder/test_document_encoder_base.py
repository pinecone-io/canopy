import pytest
from .base_test_document_encoder import BaseTestDocumentEncoder
from ..stubs.stub_dense_encoder import StubDenseEncoder
from ..stubs.stub_document_encoder import StubDocumentEncoder


class TestStubDocumentEncoder(BaseTestDocumentEncoder):

    @staticmethod
    @pytest.fixture
    def expected_dimension():
        return 3

    @staticmethod
    @pytest.fixture
    def inner_encoder(expected_dimension):
        return StubDenseEncoder(dimension=3)

    @staticmethod
    @pytest.fixture
    def document_encoder(inner_encoder):
        return StubDocumentEncoder(inner_encoder,
                                   batch_size=2)
