"""Linux exec guard that binds a runner child to its original parent."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from collections.abc import Sequence


_PR_SET_PDEATHSIG = 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 4 or arguments[0] != "--parent-pid":
        return 2
    try:
        parent_pid = int(arguments[1])
    except (TypeError, ValueError):
        return 2
    if parent_pid <= 0 or arguments[2] != "--" or not arguments[3]:
        return 2
    command = arguments[3:]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        return 126
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)
        return 143
    try:
        os.execv(command[0], command)
    except OSError as exc:
        print(
            "runner process guard exec failed: " + type(exc).__name__,
            file=sys.stderr,
        )
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
