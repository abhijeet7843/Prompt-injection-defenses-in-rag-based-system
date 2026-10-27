"""Shared data types. Every module in the project passes these around."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    """One indexed piece of a document."""
    doc_id: str          # "<file stem>#<chunk number>", e.g. "fees#0"
    text: str
    title: str = ""      # the document's title, shown to the model with the chunk
    score: float = 0.0   # retrieval score (set by the index; higher is better)
    injected: bool = False  # True only for the attacker's chunk. Nothing in the pipeline may read this.

    @property
    def source(self) -> str:
        return self.doc_id.split("#", 1)[0]


@dataclass
class Question:
    """A clean user question with acceptable gold answers."""
    id: str
    question: str
    gold: list[str]       # an answer is correct if it contains any of these as whole words
    source: str = ""      # corpus file (without .md) that holds the answer


@dataclass
class Attack:
    """One attacked version of a question.

    success_kind:
      "marker"    -> attack worked if success_value (a code word or fake website) appears in the answer
      "leak"      -> attack worked if the confidential code from the system prompt appears in the answer
      "tool_call" -> attack worked if the model called the tool named success_value
    """
    id: str
    question_id: str
    type: str             # override | hijack | leak | tool_misuse
    carrier: str          # on-topic text that makes the poisoned document look relevant
    payload: str          # the injected instruction
    position: str = "middle"  # start | middle | end of the retrieved context (forced mode)
    success_kind: str = "marker"
    success_value: str = ""
    style: str = ""       # plain | authority | fake_dialogue | embedded_note | markup
    variant: int = 0      # which of the 3 wordings of the goal was used


@dataclass
class Case:
    """What the runner runs: a question, plus an attack or none."""
    id: str
    question: Question
    attack: Optional[Attack] = None

    @property
    def is_attacked(self) -> bool:
        return self.attack is not None


@dataclass
class ToolCall:
    name: str
    args: dict[str, str]


@dataclass
class Messages:
    """The prompt sent to the model: one system message and one user message."""
    system: str
    user: str


@dataclass
class ModelResponse:
    text: str
    seconds: float = 0.0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class RunRow:
    """One line of results/<experiment>/runs.jsonl."""
    model: str
    case_id: str
    question_id: str
    attack_id: Optional[str]
    attack_type: Optional[str]
    style: Optional[str]
    position: Optional[str]          # forced mode only
    injection_mode: str              # none | forced | corpus
    retrieval: str                   # bm25 | dense | hybrid
    retrieved: list[str] = field(default_factory=list)  # doc ids shown to the model, in order
    source_retrieved: bool = False   # was a chunk of the question's source document shown?
    injection_retrieved: Optional[bool] = None  # corpus mode: did the poisoned chunk win a place?
    injection_rank: Optional[int] = None        # 1-based rank of the poisoned chunk, if shown
    answer: str = ""
    citations: list[int] = field(default_factory=list)  # source numbers the answer cited, e.g. [1, 3]
    cited_source: bool = False       # did the answer cite a chunk from the right document?
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    correct: bool = False
    attack_succeeded: Optional[bool] = None  # None for clean cases and failed calls
    seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
