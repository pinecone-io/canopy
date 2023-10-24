import pytest

from canopy.knowledge_base.record_encoder import DenseRecordEncoder
from .base_test_record_encoder import BaseTestRecordEncoder
from ..stubs.stub_dense_encoder import StubDenseEncoder


class TestStubRecordEncoder(BaseTestRecordEncoder):

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
    def record_encoder(inner_encoder):
        return DenseRecordEncoder(inner_encoder, batch_size=2)
