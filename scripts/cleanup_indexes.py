import sys
from tests.util import cleanup_indexes


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/cleanup_indexes.py <testrun_uid>")
        sys.exit(1)

    testrun_uid = sys.argv[1]
    cleanup_indexes(testrun_uid)


if __name__ == '__main__':
    main()
