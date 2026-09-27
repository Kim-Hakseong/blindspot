from blindspot.boundary.store import ProbeStore


def test_roundtrip_survives_reopen(tmp_path):
    path = tmp_path / "probes.json"
    key = ProbeStore.key("road100", 100, "yolox_s", [{"axis": "a.b", "value": 1.0}], 7)
    ProbeStore(path).put(key, {"map50": 0.5})
    assert ProbeStore(path).get(key) == {"map50": 0.5}


def test_key_is_independent_of_dict_ordering():
    a = ProbeStore.key("d", 1, "p", [{"axis": "x", "value": 1.0, "unit": "ms"}], 1)
    b = ProbeStore.key("d", 1, "p", [{"unit": "ms", "value": 1.0, "axis": "x"}], 1)
    assert a == b


def test_key_changes_with_seed_and_frames():
    base = ProbeStore.key("d", 100, "p", [], 1)
    assert base != ProbeStore.key("d", 100, "p", [], 2)
    assert base != ProbeStore.key("d", 50, "p", [], 1)
