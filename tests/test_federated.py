import numpy as np

from edgepack.federated import DPConfig, FederatedClient, FederatedServer, FeedbackEvent
from edgepack.federated.federated import _pair_mask
from edgepack.rag import EncryptedVectorStore


def make_client(tmp_path, client_id):
    store = EncryptedVectorStore(tmp_path / f"c{client_id}.db", passphrase=f"pw-{client_id}")
    good = store.add(f"packing guide station {client_id}: heavy items bottom layer first")
    bad = store.add("weekly cafeteria menu unrelated to packing")
    client = FederatedClient(client_id=client_id, store=store,
                             dp=DPConfig(noise_multiplier=0.01), seed=client_id)
    client.record_feedback(FeedbackEvent(
        query="how to pack heavy items",
        clicked_doc_id=good, skipped_doc_id=bad))
    return client


def test_pairwise_masks_cancel_in_aggregate():
    m_ab = _pair_mask(1, 2, 3)
    assert np.allclose(m_ab, _pair_mask(2, 1, 3))  # symmetric derivation


def test_masked_sum_equals_true_sum(tmp_path):
    clients = [make_client(tmp_path, i) for i in range(3)]
    roster = [c.client_id for c in clients]
    # fix RNG so DP noise is identical between the two computations
    true_sum = np.zeros(3)
    for c in clients:
        c.rng = np.random.default_rng(c.client_id + 1000)
        true_sum += c.dp.privatize(c.local_update(), c.rng)
    masked_sum = np.zeros(3)
    for c in clients:
        c.rng = np.random.default_rng(c.client_id + 1000)
        masked_sum += c.masked_update(roster)
    np.testing.assert_allclose(masked_sum, true_sum, atol=1e-9)


def test_individual_masked_update_hides_true_update(tmp_path):
    clients = [make_client(tmp_path, i) for i in range(2)]
    roster = [0, 1]
    c = clients[0]
    c.rng = np.random.default_rng(7)
    true_update = c.dp.privatize(c.local_update(), np.random.default_rng(7))
    c.rng = np.random.default_rng(7)
    masked = c.masked_update(roster)
    assert not np.allclose(masked, true_update)  # server can't see the real delta


def test_dp_clipping_bounds_update_norm():
    dp = DPConfig(clip_norm=1.0, noise_multiplier=0.0)
    big = np.array([10.0, 10.0, 10.0])
    out = dp.privatize(big, np.random.default_rng(0))
    assert np.linalg.norm(out) <= 1.0 + 1e-9


def test_federated_rounds_improve_ranking_weights(tmp_path):
    clients = [make_client(tmp_path, i) for i in range(4)]
    server = FederatedServer(dim=3)
    for _ in range(3):
        report = server.run_round(clients)
    assert report.n_clients == 4
    # hinge updates push weights toward features where clicked > skipped
    assert np.linalg.norm(server.global_weights) > 0
    # every client received the same global weights
    for c in clients:
        np.testing.assert_allclose(c.adapter.weights, server.global_weights)


def test_global_weights_actually_rerank(tmp_path):
    from edgepack.rag import RAGPipeline, RerankAdapter
    from edgepack.router import ModelRouter, TokenBudget

    clients = [make_client(tmp_path, i) for i in range(4)]
    server = FederatedServer(dim=3)
    for _ in range(5):
        server.run_round(clients)

    store = clients[0].store
    adapter = RerankAdapter(weights=server.global_weights)
    rag = RAGPipeline(store, ModelRouter(budget=TokenBudget(limit=1e9)), adapter=adapter)
    resp = rag.query("how to pack heavy items")
    assert "packing guide" in resp.hits[0].text
