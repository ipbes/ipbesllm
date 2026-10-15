"""Answer ratings (thumbs up/down from the app), one CSV row per click.

With FEEDBACK_CONTAINER_URL set (a SAS URL of an Azure Blob container, with
Read, Add, Create and Write permissions), rows are appended to the append
blob RATINGS_BLOB in that container, so they survive restarts of a hosted
app: Streamlit Community Cloud keeps no files. Download it from the portal.
Otherwise rows go to feedback/ratings.csv in the project.
"""
from __future__ import annotations

import csv
import datetime
import io
import os
from functools import lru_cache

from settings import PROJECT_ROOT

FEEDBACK_CONTAINER_URL = os.getenv("FEEDBACK_CONTAINER_URL", "")
RATINGS_BLOB = "ratings.csv"
LOCAL_PATH = PROJECT_ROOT / "feedback" / RATINGS_BLOB

FIELDS = ["timestamp", "rating", "question", "answer", "source", "route",
          "model", "sources"]


def _csv_line(values: list) -> bytes:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(values)
    return buffer.getvalue().encode("utf-8")


@lru_cache(maxsize=1)
def _ratings_blob():
    """The append blob, created with the header row on first use."""
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import ContainerClient

    blob = ContainerClient.from_container_url(
        FEEDBACK_CONTAINER_URL
    ).get_blob_client(RATINGS_BLOB)
    try:
        # Only if missing: never truncates existing ratings.
        blob.create_append_blob(if_none_match="*")
        blob.append_block(_csv_line(FIELDS))
    except ResourceExistsError:
        pass
    return blob


def save_rating(rating: str, question: str, answer: str, source: str,
                route: str | None, model: str, sources: int) -> None:
    """Append one rating ("up" or "down"). Raises if it cannot be stored."""
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
                     .isoformat(timespec="seconds"),
        "rating": rating, "question": question, "answer": answer,
        "source": source, "route": route or "", "model": model,
        "sources": sources,
    }
    values = [row[field] for field in FIELDS]

    if FEEDBACK_CONTAINER_URL:
        _ratings_blob().append_block(_csv_line(values))
        return

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOCAL_PATH.exists()
    with LOCAL_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow(values)
