import pytest
from .base_test_record_encoder import BaseTestRecordEncoder
from ..stubs.stub_dense_encoder import StubDenseEncoder
from ..stubs.stub_record_encoder import StubRecordEncoder


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
        return StubRecordEncoder(inner_encoder,
                                 batch_size=2)
