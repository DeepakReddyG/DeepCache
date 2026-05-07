import hashlib


class _HashFallback:
    def __init__(self, algorithm, data=b"", digest_size=None):
        self._hash = algorithm()
        if data:
            self.update(data)
        self._digest_size = digest_size

    def update(self, data):
        self._hash.update(data)

    def digest(self):
        digest = self._hash.digest()
        if self._digest_size is not None:
            return digest[: self._digest_size]
        return digest

    def hexdigest(self):
        return self.digest().hex()

    def copy(self):
        clone = self.__class__.__new__(self.__class__)
        clone._hash = self._hash.copy()
        clone._digest_size = self._digest_size
        return clone


if not hasattr(hashlib, "blake2b"):
    hashlib.blake2b = lambda data=b"", digest_size=64, **_: _HashFallback(  # type: ignore[attr-defined]
        hashlib.sha512, data=data, digest_size=digest_size
    )

if not hasattr(hashlib, "blake2s"):
    hashlib.blake2s = lambda data=b"", digest_size=32, **_: _HashFallback(  # type: ignore[attr-defined]
        hashlib.sha256, data=data, digest_size=digest_size
    )
