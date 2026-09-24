from dataclasses import dataclass, field
from typing import Any


@dataclass
class Source:
    source_type: str
    source_file: str
    location: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    retrieval_time: float = 0.0
    generation_time: float = 0.0