import logging
from datetime import datetime

import pinecone

logger = logging.getLogger(__name__)


def create_index_name(testrun_uid: str, prefix: str) -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    return f"{prefix}-{testrun_uid[-6:]}-{today}"


def create_system_tests_index_name(testrun_uid: str) -> str:
    return create_index_name(testrun_uid, "test-kb")


def create_e2e_tests_index_name(testrun_uid: str) -> str:
    return create_index_name(testrun_uid, "test-app")


def cleanup_indexes(testrun_uid: str):
    pinecone.init()
    e2e_index_name = create_e2e_tests_index_name(testrun_uid)
    system_index_name = create_system_tests_index_name(testrun_uid)
    index_names = (system_index_name, e2e_index_name)
    logger.info(f"Preparing to cleanup indexes: {index_names}")
    current_indexes = pinecone.list_indexes()
    for index_name in index_names:
        if index_name in current_indexes:
            logger.info(f"Deleting index: {index_name}")
            pinecone.delete_index(index_name)
        else:
            logger.info(f"Index: {index_name} already deleted.")
