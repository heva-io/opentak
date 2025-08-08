# mypy: disable-error-code="attr-defined, method-assign"
"""Logger module initializing a logger named 'tak'.

By default it will try to load a logging config file named 'logging.conf' or 'src/logging.conf'.
However if root logger is already initialized, the module does nothing else
than create a 'tak' logger.
"""

import logging
import logging.config
import os

# Create custom level
logging.INFO_HEVA = logging.INFO + 5
logging.DEBUG_HEVA = logging.DEBUG + 5
logging.addLevelName(logging.INFO_HEVA, "INFO_HEVA")
logging.addLevelName(logging.DEBUG_HEVA, "DEBUG_HEVA")


class HevaLogger(logging.getLoggerClass()):
    """Custom logger class with additional HEVA-specific logging methods."""

    def info_heva(self, msg: str, *args, **kwargs):
        """Log a message with severity 'INFO_HEVA'.

        :param msg: message to log
        :param args: arguments for message formatting
        :param kwargs: keyword arguments for the logging call
        """
        self.log(logging.INFO_HEVA, msg, *args, **kwargs)

    def debug_heva(self, msg: str, *args, **kwargs):
        """Log a message with severity 'DEBUG_HEVA'.

        :param msg: message to log
        :param args: arguments for message formatting
        :param kwargs: keyword arguments for the logging call
        """
        self.log(logging.DEBUG_HEVA, msg, *args, **kwargs)


logging.setLoggerClass(HevaLogger)

# Create package logger
logger = logging.getLogger("tak")

# Skip configuration if logging is already initialized
if not logging.getLogger().hasHandlers():
    # Load config from config file
    config_loaded = False
    for location in ["logging.conf", "src/logging.conf"]:
        if os.path.isfile(location):
            logging.config.fileConfig(location, disable_existing_loggers=False)
            config_loaded = True
            break
    if not config_loaded:
        logging.basicConfig()

    # Set log level from environment
    if "LOG_LEVEL" in os.environ:
        logger.setLevel(logging.__getattribute__(os.environ["LOG_LEVEL"]))
