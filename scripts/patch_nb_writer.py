from pathlib import Path

path = Path('scripts/write_kaggle_notebook.py')
text = path.read_text(encoding='utf-8')
text = text.replace('"execution_count": null', '"execution_count": None')
path.write_text(text, encoding='utf-8')
print('patched', path)
