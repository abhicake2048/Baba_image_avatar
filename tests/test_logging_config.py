from __future__ import annotations

from identity_generator.logging_config import configure_logging, get_logger, tail_log


def test_rotating_logger_and_tail(tmp_path) -> None:
    log_path = configure_logging(tmp_path / "logs", "INFO")
    get_logger().info("test.event prompt=%s model=%s count=%s", "red shirt", "test/model", 2)
    text = tail_log(log_path)
    assert log_path.name == "app.log"
    assert "test.event" in text
    assert "prompt=red shirt" in text
    assert "model=test/model" in text
    assert "count=2" in text
