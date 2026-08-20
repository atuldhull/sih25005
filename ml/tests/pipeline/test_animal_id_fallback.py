"""Regression test for the animal_id key-fallback fix (Rev 2 audit item 3 /
implementation item 5).

server/main.py passes the raw MongoDB animal document straight through as
animal_record. server/seed.py stores animal documents keyed by "_id", not
"animal_id" - so animal_record.get("animal_id") always returned None for
every real server-backed record. pipeline.py must fall back to "_id" when
"animal_id" is absent.
"""

import tempfile

from ml.pipeline import score_animal


def _dummy_image_path():
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.close()
    return f.name


def test_animal_id_falls_back_to_mongo_id_key():
    """Real server records only carry "_id" (server/seed.py), not "animal_id"."""
    img = _dummy_image_path()
    animal_record = {"_id": "356279812345", "species": "cattle", "breed": "Gir"}
    result = score_animal(img, img, None, animal_record)
    assert result["animal_id"] == "356279812345"


def test_animal_id_prefers_explicit_key_when_present():
    img = _dummy_image_path()
    animal_record = {"animal_id": "A123", "_id": "should_not_be_used", "species": "cattle"}
    result = score_animal(img, img, None, animal_record)
    assert result["animal_id"] == "A123"


def test_animal_id_is_none_when_neither_key_present():
    img = _dummy_image_path()
    animal_record = {"species": "cattle"}
    result = score_animal(img, img, None, animal_record)
    assert result["animal_id"] is None
