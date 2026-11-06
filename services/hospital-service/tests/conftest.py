import pathlib
import sys

# main.py and the fixture helper are imported by bare name, as the service runs them.
_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
