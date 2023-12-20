import sys
from tests.util import cleanup_indexes


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/cleanup_indexes.py <testrun_uid>")
        sys.exit(1)

    testrun_uid = sys.argv[1]
    if testrun_uid:
        print(f"Cleaning up indexes for testrun_uid: {testrun_uid}")
        cleanup_indexes(testrun_uid)
    else:
        print("testrun_uid is not passed, index cleanup will not be run.")


if __name__ == '__main__':
    main()
