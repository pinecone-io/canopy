import logging
from datetime import datetime
from typing import List

from canopy.knowledge_base.knowledge_base import _get_global_client, INDEX_NAME_PREFIX

logger = logging.getLogger(__name__)


def create_index_name(testrun_uid: str, prefix: str) -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    return f"{testrun_uid[-6:]}-{prefix}-{today}"


def create_system_tests_index_name(testrun_uid: str) -> str:
    return create_index_name(testrun_uid, "test-kb")


def create_e2e_tests_index_name(testrun_uid: str, index_type: str) -> str:
    return create_index_name(testrun_uid, f"test-app-{index_type}")


def get_related_indexes(indexes: List[str], testrun_uid: str) -> List[str]:
    return [
        index for index in indexes
        if index.startswith(f"{INDEX_NAME_PREFIX}{testrun_uid[-6:]}")
    ]


def cleanup_indexes(testrun_uid: str):
    client = _get_global_client()
    current_indexes = client.list_indexes().names()
    index_names = get_related_indexes(current_indexes, testrun_uid)
    logger.info(f"Preparing to cleanup indexes: {index_names}")
    for index_name in index_names:
        logger.info(f"Deleting index '{index_name}'...")
        client.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted.")
