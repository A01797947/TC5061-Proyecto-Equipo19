"""Arranque del worker RQ: precarga los modelos y consume la cola de predicciones."""
from __future__ import annotations

import logging
import os

from redis import Redis
from rq import Queue, Worker

from deployment.api.jobs import QUEUE_NAME, get_store


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    get_store()  # precarga en el proceso padre; los hijos del fork la heredan
    connection = Redis.from_url(os.environ["REDIS_URL"])
    Worker([Queue(QUEUE_NAME, connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()