"""At-rest encryption for the on-device vector database.

AES-256-GCM (authenticated encryption) with a key derived from a device
passphrase via scrypt.  Every record gets a fresh random 96-bit nonce, and the
record ID is bound in as *associated data* so ciphertexts cannot be swapped
between rows without detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

KEY_LEN = 32          # AES-256
NONCE_LEN = 12        # GCM standard
SALT_LEN = 16
SCRYPT_N = 2 ** 14    # interactive-grade work factor for edge CPUs
SCRYPT_R = 8
SCRYPT_P = 1


class DecryptionError(Exception):
    """Wrong key, or the ciphertext/associated data was tampered with."""


def derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def new_salt() -> bytes:
    return os.urandom(SALT_LEN)


@dataclass
class EncryptedBlob:
    nonce: bytes
    ciphertext: bytes  # includes the GCM tag

    def pack(self) -> bytes:
        return self.nonce + self.ciphertext

    @classmethod
    def unpack(cls, blob: bytes) -> "EncryptedBlob":
        return cls(nonce=blob[:NONCE_LEN], ciphertext=blob[NONCE_LEN:])


class RecordCipher:
    """Encrypts/decrypts individual DB records with per-record nonces."""

    def __init__(self, key: bytes):
        if len(key) != KEY_LEN:
            raise ValueError(f"key must be {KEY_LEN} bytes")
        self._aead = AESGCM(key)

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes) -> "RecordCipher":
        return cls(derive_key(passphrase, salt))

    def encrypt(self, plaintext: bytes, record_id: str) -> bytes:
        nonce = os.urandom(NONCE_LEN)
        ct = self._aead.encrypt(nonce, plaintext, record_id.encode("utf-8"))
        return EncryptedBlob(nonce, ct).pack()

    def decrypt(self, blob: bytes, record_id: str) -> bytes:
        eb = EncryptedBlob.unpack(blob)
        try:
            return self._aead.decrypt(eb.nonce, eb.ciphertext, record_id.encode("utf-8"))
        except InvalidTag as exc:
            raise DecryptionError(f"record {record_id!r}: bad key or tampered data") from exc
