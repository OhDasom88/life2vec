from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Generic, List, NewType, Optional, Sequence, TypeVar

JSONSerializable = NewType("JSONSerializable", object)

if TYPE_CHECKING:
    from src.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class PersonDocument:
    """Dataclass for defining the complete person document in a structured fashion"""

    person_id: int
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]
    timecut_pos: int  # What is this?
    segment: Optional[List[int]] = None
    background: Optional["Background"] = None
    shuffled: bool = False
    task_info: Optional[JSONSerializable] = None
    # Optional event-level metadata.  Legacy corpora can leave every field unset.
    event_ids: Optional[List[Any]] = None
    same_time_group_ids: Optional[List[Any]] = None
    event_kinds: Optional[List[Any]] = None
    modality_refs: Optional[List[Any]] = None
    order_semantics: Optional[str] = None
    op_eligible: Optional[bool] = None

    def event_aligned_fields(self) -> List[str]:
        """Return fields whose i-th value describes the i-th sentence/event."""
        fields = ["sentences", "abspos", "age"]
        for name in (
            "segment",
            "event_ids",
            "same_time_group_ids",
            "event_kinds",
            "modality_refs",
        ):
            if getattr(self, name) is not None:
                fields.append(name)
        return fields

    def validate_event_alignment(self) -> None:
        """Fail early instead of silently training on misaligned event metadata."""
        expected = len(self.sentences)
        mismatched = {
            name: len(getattr(self, name))
            for name in self.event_aligned_fields()
            if len(getattr(self, name)) != expected
        }
        if mismatched:
            raise ValueError(
                f"Event-aligned fields must have length {expected}: {mismatched}"
            )

    def select_events(self, indices: Sequence[int]) -> "PersonDocument":
        """Apply one selection/permutation to every event-aligned field in-place."""
        self.validate_event_alignment()
        order = [int(i) for i in indices]
        for name in self.event_aligned_fields():
            values = getattr(self, name)
            setattr(self, name, [values[i] for i in order])
        self.validate_event_alignment()
        return self


@dataclass
class Background:
    """Defines the background information about a person"""

    origin: str
    gender: str
    birth_month: int
    birth_year: int

    @staticmethod
    def get_sentence(x: Optional["Background"]) -> List[str]:
        """Return sequence of tokens corresponding to this person. Implemented as
        classmethod since we can null the background in PersonDocument in case of
        unknown background.
        """

        if x is None:
            return 4 * ["[UNK]"]
        else:
            return [
                x.origin,
                x.gender,
                f"MONTH_{x.birth_month}",
                f"YEAR_{x.birth_year}",
            ]


class EncodedDocument(Generic[_TaskT]):
    """Generic class for encoded documents. Each task can then type-hint their
    specific encoding using a dataclass like

    .. code-block ::

        class MyTask:
            def encode_document(x: PersonDocument) -> "MyTaskEncodedDocument":
                return MyTaskEncodedDocument(target=1)

        @dataclass
        class MyTaskEncodedDocument(EncodedDocument[MyTask]):
            target: int

    """
