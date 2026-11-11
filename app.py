from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from identity_generator.config import load_settings
from identity_generator.logging_config import configure_logging, get_logger
from identity_generator.runtime import build_runtime
from identity_generator.ui import CSS, THEME, build_app


def main() -> int:
    project_root = Path(__file__).resolve().parent
    try:
        settings = load_settings(project_root)
        configure_logging(settings.log_dir, settings.log_level)
        get_logger().info("app.starting host=127.0.0.1 port=7860")
        runtime = asyncio.run(build_runtime(settings))
    except (ValidationError, ValueError, RuntimeError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        print("See README.md for setup instructions.", file=sys.stderr)
        return 1

    demo = build_app(runtime)
    try:
        demo.queue(default_concurrency_limit=1, max_size=4).launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            allowed_paths=[str(settings.output_dir)],
            inbrowser=True,
            css=CSS,
            theme=THEME,
        )
    finally:
        get_logger().info("app.stopping")
        asyncio.run(runtime.close())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

