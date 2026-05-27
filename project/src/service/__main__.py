from __future__ import annotations

import uvicorn

from src.config import load_config


def main() -> None:
    config = load_config()
    uvicorn.run("src.service.app:app", host=config.service_host, port=config.service_port, reload=False)


if __name__ == "__main__":
    main()

