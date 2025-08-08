import importlib
import logging
import os
from unittest.mock import patch

import pytest

import opentak.logger
from opentak.logger import logger


def test_log_levels():
    assert (
        logging.DEBUG
        < logging.DEBUG_HEVA
        < logging.INFO
        < logging.INFO_HEVA
        < logging.WARNING
    )


def test_logger(caplog):
    # Given
    caplog.set_level(logging.INFO_HEVA)

    # When
    logger.info("A")
    logger.info_heva("B")
    logger.warning("C")

    # Then
    assert caplog.messages == ["B", "C"]


@patch("logging.Logger.hasHandlers")
def test_log_level(has_handlers, caplog):
    # Given
    has_handlers.return_value = False
    os.environ["LOG_LEVEL"] = "DEBUG_HEVA"

    # When
    logger.debug_heva("A")
    importlib.reload(opentak.logger)
    logger.debug_heva("B")

    # Then
    assert logger.level == logging.DEBUG_HEVA
    assert caplog.messages == ["B"]


@pytest.fixture
def logging_conf():
    with open("logging.conf", "w") as f:
        f.write("""
[handlers]
keys=console

[formatters]
keys=default

[loggers]
keys=root

[formatter_default]
format=CustomFmt %(levelname)s:%(message)s

[handler_console]
class=logging.StreamHandler
level=NOTSET
formatter=default
args=(sys.stdout,)

[logger_root]
handlers=console
propagate=0
        """)
    yield
    os.remove("logging.conf")


@patch("logging.Logger.hasHandlers")
def test_auto_config(has_handlers, logging_conf):
    # Given
    has_handlers.return_value = False

    # When
    importlib.reload(opentak.logger)

    # Then
    assert len(logging.getLogger().handlers) == 1
    assert isinstance(logging.getLogger().handlers[0], logging.StreamHandler)
