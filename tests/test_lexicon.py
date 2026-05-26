import json
from pathlib import Path


def test_lexicon_file_exists():
    p = Path('data/lexicon.json')
    assert p.exists(), 'data/lexicon.json must exist'


def test_lexicon_valid_json():
    p = Path('data/lexicon.json')
    data = json.loads(p.read_text(encoding='utf-8'))
    assert isinstance(data, dict), 'lexicon must be a dict'


def test_lexicon_labels_present():
    p = Path('data/lexicon.json')
    data = json.loads(p.read_text(encoding='utf-8'))
    required = {'Happy', 'Sad', 'Toxic', 'Funny', 'Neutral'}
    assert required.issubset(set(data.keys())), f'missing labels: {required - set(data.keys())}'


def test_lexicon_nonempty_entries():
    p = Path('data/lexicon.json')
    data = json.loads(p.read_text(encoding='utf-8'))
    for k, v in data.items():
        assert isinstance(v, list), f'entry {k} must be a list'
        assert len(v) > 0, f'entry {k} must not be empty'
