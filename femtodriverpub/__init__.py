def _get_dir():
    import pathlib
    return pathlib.Path(__file__).parent.resolve()

__version__ = (_get_dir() / 'VERSION').read_text(encoding='utf-8').strip()

from .spu_runner import SPURunner, FakeSPURunner
