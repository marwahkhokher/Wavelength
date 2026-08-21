import sys
from pathlib import Path

# persona-generation/ is hyphenated and can't be imported as a real Python
# package (see the note in generation_prompt.py) - every module here uses
# flat sibling imports, so tests need this directory on sys.path directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
