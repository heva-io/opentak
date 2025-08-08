import logging

class HevaLogger(logging.Logger):
    def info_heva(self, msg: str, *args, **kwargs): ...
    def debug_heva(self, msg: str, *args, **kwargs): ...

logger: HevaLogger
