"""Package release artifacts into a timestamped release/ directory and zip file.

Usage: python scripts/package_release.py
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
import zipfile


def gather_files(release_dir: Path):
    files_to_copy = [
        ('models', True),
        ('data/lexicon.json', False),
        ('src/prediction.py', False),
        ('src/data_preprocessing.py', False),
        ('requirements.txt', False),
        ('tests', True),
    ]

    for path, is_dir in files_to_copy:
        src = Path(path)
        dst = release_dir / src.name
        if not src.exists():
            print(f'Warning: {src} does not exist, skipping')
            continue
        if is_dir:
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_run_example(release_dir: Path):
    run_code = '''from prediction import SentimentPredictor

predictor = SentimentPredictor(model_path='models/best_model.joblib',
                               vectorizer_path='models/vectorizer.joblib',
                               label_encoder_path='models/label_encoder.joblib')

text = 'আমি খুব খুশি'
label, conf = predictor.predict(text)
print(text, '->', label, conf)
'''
    (release_dir / 'run_predict.py').write_text(run_code, encoding='utf-8')


def make_zip(release_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(release_dir):
            for f in files:
                full = Path(root) / f
                zf.write(full, full.relative_to(release_dir.parent))


def main():
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    release_root = Path('release')
    release_root.mkdir(exist_ok=True)
    release_dir = release_root / f'release-{ts}'
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir()

    gather_files(release_dir)
    write_run_example(release_dir)

    zip_path = release_root / f'{release_dir.name}.zip'
    if zip_path.exists():
        zip_path.unlink()
    make_zip(release_dir, zip_path)
    print(f'Created release at {release_dir} and {zip_path}')


if __name__ == '__main__':
    main()
