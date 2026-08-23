import app as note_app


def test_index_renders():
    client = note_app.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Clinical Note Generator" in resp.data


def test_generate_note_requires_dialogue():
    client = note_app.app.test_client()
    resp = client.post("/generate_note", json={})
    assert resp.status_code == 400


def test_decode_matches_eval_protocol():
    gc = note_app.generation_config
    assert gc.do_sample is False
    assert gc.num_beams == 4
    assert gc.repetition_penalty == 2.0
    assert gc.no_repeat_ngram_size == 4


def test_input_length_matches_training():
    assert note_app.INPUT_MAX_LENGTH == 512


def test_is_boilerplate():
    assert note_app.is_boilerplate("Noncontributory.")
    assert note_app.is_boilerplate("None.")
    assert not note_app.is_boilerplate("The patient is a 54-year-old female with facial twitching.")
