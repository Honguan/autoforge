"""Pack errors carry a stable code so AI clients can branch on them."""

from __future__ import annotations

from autoforge.errors import AutoForgeError


class PackError(AutoForgeError):
    code = "PACK_ERROR"


class PackInvalid(PackError):
    code = "PACK_INVALID"
    hint = "Run `autoforge pack validate <path> --json` to see every problem."

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems), details={"problems": problems})


class PackNotFound(PackError):
    code = "PACK_NOT_FOUND"
    hint = "Use `autoforge pack search` or `autoforge pack list` to see available packs."


class PackExists(PackError):
    code = "PACK_EXISTS"
    hint = "Pass --force to reinstall."


class PackSourceError(PackError):
    code = "PACK_SOURCE_ERROR"


class ShaMismatch(PackError):
    code = "SHA_MISMATCH"
    hint = "The source changed after it was indexed. Do not install; report it to the index maintainer."


class IndexUnavailable(PackError):
    code = "INDEX_UNAVAILABLE"
    hint = "Check the network or set AUTOFORGE_INDEX_URL to a local index.json."
