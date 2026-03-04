# Prevent pytest from collecting synthetic Python files in the fixture directories.
# These files are demo/test data (e.g. jwt.py, test_auth.py) that are not real test modules.
collect_ignore_glob = ["recovery/**/*.py"]
