import pytest

from defence_rag.parsing import extract_json, get_bool, get_int_list, get_str


def test_extract_json_plain():
    assert extract_json('{"route": "vectorstore"}') == {"route": "vectorstore"}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_text():
    assert extract_json('Here is my answer:\n{"route": "vectorstore", "reason": "x"} thanks') == {
        "route": "vectorstore",
        "reason": "x",
    }


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_get_bool_variants():
    assert get_bool({"supported": True}, ["supported"]) is True
    assert get_bool({"supported": "true"}, ["supported"]) is True
    assert get_bool({"useful": "no"}, ["useful"]) is False
    assert get_bool({}, ["x"]) is False
    assert get_bool({"a": "yes"}, ["b", "a"]) is True


def test_get_str_first_nonempty_key():
    assert get_str({"a": "", "b": "NATO"}, ["a", "b"]) == "NATO"


def test_get_int_list():
    assert get_int_list({"relevant_indices": [0, "2", 3.0]}, ["relevant_indices"]) == [0, 2, 3]
    assert get_int_list({"relevant_indices": "not-a-list"}, ["relevant_indices"]) == []


def test_hybrid_search_rrf_and_limit(seeded_store):
    from defence_rag.llm import CachedEmbeddings

    class Dummy(CachedEmbeddings):
        def embed_query(self, text):
            return [1.0] * 384

    results = seeded_store.hybrid_search("What does NATO do?", Dummy().embed_query("x"), k=2)
    assert len(results) == 2
    sources = {d.metadata["source"] for d in results}


def test_count_and_points_roundtrip(seeded_store):
    assert seeded_store.count == 2
    points = seeded_store.client.scroll(seeded_store.collection, limit=10)[0]
    payload = points[0].payload
    assert payload["text"] in {
        "NATO relies on collective defence and deterrence against Russia.",
        "The US Air Force executes air superiority and long-range strike missions.",
    }