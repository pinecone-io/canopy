import pytest

TEST_NAMESPACE = "ns"
TEST_CREATE_INDEX_PARAMS = [
    {"spec": {"serverless": {"cloud": "aws", "region": "us-west-2"}}},
    {"spec": {"pod": {"environment": "eu-west1-gcp", "pod_type": "p1.x1"}}},
    {"spec": {"pod": {"environment": "gcp-starter", "pod_type": "p1.x1"}}},
]


@pytest.fixture(scope="module", params=[None, TEST_NAMESPACE])
def namespace(request):
    return request.param


@pytest.fixture(scope="module",
                params=TEST_CREATE_INDEX_PARAMS,
                # The first key in the spec is the index type ("serverless" \ "pod")
                ids=[next(iter(_["spec"])) for _ in TEST_CREATE_INDEX_PARAMS])
def create_index_params(request):
    return request.param
