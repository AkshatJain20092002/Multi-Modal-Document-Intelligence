import logging

import pytest

from app.observability.operation_logger import log_operation


def test_log_operation_logs_start_and_success(caplog):
    with caplog.at_level(logging.INFO):
        with log_operation("do_thing", foo="bar"):
            pass
    messages = [r.getMessage() for r in caplog.records]
    assert any("operation.start op=do_thing" in m for m in messages)
    assert any("operation.success op=do_thing" in m for m in messages)


def test_log_operation_logs_failure_and_reraises(caplog):
    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            with log_operation("do_thing"):
                raise ValueError("boom")
    messages = [r.getMessage() for r in caplog.records]
    assert any("operation.failed op=do_thing" in m for m in messages)


def test_log_operation_as_decorator(caplog):
    @log_operation("decorated_op")
    def f(x):
        return x * 2

    with caplog.at_level(logging.INFO):
        assert f(21) == 42
    messages = [r.getMessage() for r in caplog.records]
    assert any("operation.success op=decorated_op" in m for m in messages)
