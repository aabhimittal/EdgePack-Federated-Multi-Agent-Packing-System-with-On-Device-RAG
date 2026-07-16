import numpy as np
import pytest

from edgepack.crypto import DecryptionError, RecordCipher, derive_key, new_salt
from edgepack.rag import EncryptedVectorStore, HashingEmbedder


def make_cipher(passphrase="test-pass"):
    return RecordCipher.from_passphrase(passphrase, salt=b"0" * 16)


def test_encrypt_decrypt_roundtrip():
    c = make_cipher()
    blob = c.encrypt(b"secret payload", "rec-1")
    assert c.decrypt(blob, "rec-1") == b"secret payload"
    assert blob != b"secret payload"


def test_wrong_key_fails():
    blob = make_cipher("right").encrypt(b"data", "rec-1")
    with pytest.raises(DecryptionError):
        make_cipher("wrong").decrypt(blob, "rec-1")


def test_record_id_binding_prevents_swapping():
    c = make_cipher()
    blob = c.encrypt(b"data", "rec-1")
    with pytest.raises(DecryptionError):
        c.decrypt(blob, "rec-2")  # ciphertext moved to another row


def test_nonces_are_unique_per_encryption():
    c = make_cipher()
    assert c.encrypt(b"x", "r") != c.encrypt(b"x", "r")


def test_key_derivation_is_deterministic():
    salt = new_salt()
    assert derive_key("pw", salt) == derive_key("pw", salt)
    assert derive_key("pw", salt) != derive_key("pw2", salt)


# ---------------------------------------------------------------- store
def test_store_add_search_roundtrip(tmp_path):
    store = EncryptedVectorStore(tmp_path / "v.db", passphrase="pw")
    store.add("the packing robot places heavy boxes at the bottom")
    store.add("the cafeteria serves lunch at noon")
    hits = store.search("where do heavy boxes go", k=1)
    assert len(hits) == 1
    assert "heavy boxes" in hits[0].text


def test_store_data_is_encrypted_at_rest(tmp_path):
    path = tmp_path / "v.db"
    store = EncryptedVectorStore(path, passphrase="pw")
    store.add("SUPERSECRETTOKEN inside the document")
    store.close()
    raw = path.read_bytes()
    assert b"SUPERSECRETTOKEN" not in raw


def test_store_reopen_with_correct_passphrase(tmp_path):
    path = tmp_path / "v.db"
    s1 = EncryptedVectorStore(path, passphrase="pw")
    doc_id = s1.add("persistent document about thermal budgets")
    s1.close()

    s2 = EncryptedVectorStore(path, passphrase="pw")
    assert len(s2) == 1
    text, _ = s2.get(doc_id)
    assert "thermal budgets" in text
    assert s2.search("thermal budget", k=1)[0].doc_id == doc_id


def test_store_reopen_with_wrong_passphrase_fails(tmp_path):
    path = tmp_path / "v.db"
    s1 = EncryptedVectorStore(path, passphrase="pw")
    s1.add("doc")
    s1.close()
    with pytest.raises(DecryptionError):
        EncryptedVectorStore(path, passphrase="not-pw")


def test_store_delete(tmp_path):
    store = EncryptedVectorStore(tmp_path / "v.db", passphrase="pw")
    a = store.add("first doc about packing")
    store.add("second doc about packing")
    store.delete(a)
    assert len(store) == 1
    assert all(h.doc_id != a for h in store.search("packing", k=5))


def test_embedder_similarity_ordering():
    e = HashingEmbedder()
    q = e.embed("stable box placement")
    close = e.embed("placement of a stable box")
    far = e.embed("chocolate cake recipe with strawberries")
    assert np.dot(q, close) > np.dot(q, far)
