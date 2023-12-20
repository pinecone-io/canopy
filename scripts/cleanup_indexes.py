import logging
import sys
from tests.util import cleanup_indexes


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) != 2:
        logger.info("Usage: python scripts/cleanup_indexes.py <testrun_uid>")
        sys.exit(1)

    testrun_uid = sys.argv[1]
    if testrun_uid:
        logger.info(f"Cleaning up indexes for testrun_uid '{testrun_uid}'")
        cleanup_indexes(testrun_uid)
    else:
        logger.info("Passed testrun_uid is empty.")


if __name__ == '__main__':
    main()
