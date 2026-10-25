"""Test-run guards."""

import sys

# Under -O every assert is stripped, so a run would report success without
# testing anything.
if sys.flags.optimize:
    raise RuntimeError("run the tests without -O; assertions are stripped")
