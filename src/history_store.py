"""Chat history: each user's conversations, listed in the app's sidebar.

A user is known by the e-mail address the host gives (st.user.email:
Streamlit Community Cloud sets it for viewers of a private app). Their
conversations are kept under history/<hash of the address>/: index.json
lists them (id, title, last update) and <id>.json holds each one's
messages, sources included. Only the index is read to draw the list.

With FEEDBACK_CONTAINER_URL set, the files are blobs in that container
(the ratings' SAS permissions are enough: no listing is needed); otherwise
they go to history/ in the project.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from functools import lru_cache

from feedback_store import FEEDBACK_CONTAINER_URL
from settings import PROJECT_ROOT

LOCAL_DIR = PROJECT_ROOT / "history"
MAX_CONVERSATIONS = 50          # oldest ones drop out of the index
TITLE_LENGTH = 60


def user_folder(email: str) -> str:
    """The user's folder: a hash, so addresses are not stored in names."""
    return "history/" + hashlib.sha256(email.strip().lower().encode()).hexdigest()[:24]


@lru_cache(maxsize=1)
def _container():
    from azure.storage.blob import ContainerClient
    return ContainerClient.from_container_url(FEEDBACK_CONTAINER_URL)


def _read(path: str):
    """The JSON at `path`, or None if there is none."""
    if FEEDBACK_CONTAINER_URL:
        from azure.core.exceptions import ResourceNotFoundError
        try:
            data = _container().get_blob_client(path).download_blob().readall()
        except ResourceNotFoundError:
            return None
        return json.loads(data)
    file = PROJECT_ROOT / path
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def _write(path: str, value) -> None:
    data = json.dumps(value, ensure_ascii=False, default=_plain)
    if FEEDBACK_CONTAINER_URL:
        _container().get_blob_client(path).upload_blob(
            data.encode("utf-8"), overwrite=True,
            content_type="application/json")
        return
    file = PROJECT_ROOT / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(data, encoding="utf-8")


def _delete(path: str) -> None:
    if FEEDBACK_CONTAINER_URL:
        from azure.core.exceptions import ResourceNotFoundError
        try:
            _container().get_blob_client(path).delete_blob()
        except ResourceNotFoundError:
            pass
        return
    (PROJECT_ROOT / path).unlink(missing_ok=True)


def _plain(value):
    """numpy numbers and arrays (Chroma distances) as plain JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def title_for(question: str) -> str:
    question = " ".join(question.split())
    if len(question) <= TITLE_LENGTH:
        return question
    return question[:TITLE_LENGTH - 1].rsplit(" ", 1)[0] + "…"


def list_conversations(folder: str) -> list[dict]:
    """[{"id", "title", "updated"}], most recent first."""
    return _read(f"{folder}/index.json") or []


def load_conversation(folder: str, conversation_id: str) -> list[dict]:
    return (_read(f"{folder}/{conversation_id}.json") or {}).get("messages", [])


def save_conversation(folder: str, conversation_id: str,
                      messages: list[dict]) -> list[dict]:
    """Store the messages and move the conversation to the top of the
    index. Returns the new index."""
    _write(f"{folder}/{conversation_id}.json", {"messages": messages})
    index = [entry for entry in list_conversations(folder)
             if entry["id"] != conversation_id]
    first_question = next(m["content"] for m in messages if m["role"] == "user")
    index.insert(0, {
        "id": conversation_id,
        "title": title_for(first_question),
        "updated": datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec="seconds"),
    })
    for dropped in index[MAX_CONVERSATIONS:]:
        _delete(f"{folder}/{dropped['id']}.json")
    index = index[:MAX_CONVERSATIONS]
    _write(f"{folder}/index.json", index)
    return index


def delete_conversation(folder: str, conversation_id: str) -> list[dict]:
    index = [entry for entry in list_conversations(folder)
             if entry["id"] != conversation_id]
    _write(f"{folder}/index.json", index)
    _delete(f"{folder}/{conversation_id}.json")
    return index
