import eval as app_module


def test_index_renders():
    client = app_module.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Clinical Note Summarizer" in resp.data


def test_generate_requires_fields():
    client = app_module.app.test_client()
    resp = client.post("/generate", json={})
    assert resp.status_code == 400


def test_decode_matches_eval_protocol():
    gc = app_module.generation_config
    assert gc.do_sample is False
    assert gc.num_beams == 4
    assert gc.repetition_penalty == 2.0
    assert gc.no_repeat_ngram_size == 4
