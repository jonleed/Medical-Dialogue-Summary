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
