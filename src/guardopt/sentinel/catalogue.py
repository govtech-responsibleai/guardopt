"""How to describe a set of guardrails whose score semantics you have verified.

**This package ships no catalogue of its own, and that is deliberate.** What a guardrail's
score means is a property of *your* guardrail deployment, not of this library. Two
installations of the same detector can differ in version, in calibration, and in which
direction of the score means "riskier". A catalogue shipped here would be a claim about
someone else's system that this package cannot check.

So the catalogue is a type you populate, not a list you inherit. Build one from your own
observations and pass it in.

**A catalogue is advisory, not a whitelist.** The optimiser never checks a name against
one. Restricting it would stop a team evaluating a guardrail nobody has verified yet,
which is exactly the situation in which measuring it is most useful. What a catalogue IS:
the set whose semantics someone has actually checked, so a UI can offer them without
asking a user to supply a score direction.

**The optimiser never infers direction.** A guessed direction silently inverts every
threshold that guardrail contributes, producing a wrong answer that looks like a right
one. Every entry states its direction; nothing here deduces one.

**Exclusions are recorded with their evidence, not dropped silently.** "We looked at this
and it did not work" is more useful to the next person than absence — and it stops the
same guardrail being re-added, re-tested and re-rejected every six months. An excluded
guardrail is not blocked: a caller can still supply it explicitly.

Both of those principles are enforced by the type rather than left to discipline — see the
validation in `GuardrailCatalogue`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from guardopt.domain.inputs import GuardrailDefinition

__all__ = [
    "EMPTY_CATALOGUE",
    "GuardrailCatalogue",
]

_NO_ENTRIES: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class GuardrailCatalogue:
    """Guardrails you have verified, with where each claim came from.

    `provenance` answers "how do you know?" for every offered guardrail, and `exclusions`
    answers "why not this one?" for the ones you considered and rejected. Both are
    required to be non-empty strings where present, because an empty reason is worse than
    no reason: it looks like the question was answered.
    """

    entries: tuple[GuardrailDefinition, ...] = ()

    #: Guardrail name -> how its semantics were established. Every entry needs one.
    provenance: Mapping[str, str] = field(default=_NO_ENTRIES)

    #: Guardrail name -> the observation that excluded it.
    exclusions: Mapping[str, str] = field(default=_NO_ENTRIES)

    def __post_init__(self) -> None:
        names = [definition.name for definition in self.entries]

        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"catalogue lists the same guardrail more than once: {sorted(duplicates)}"
            )

        # Every offered guardrail must say how its semantics were established. Without
        # this, an entry someone added on a hunch is indistinguishable from one verified
        # against live responses — and the whole point of a catalogue is that a caller can
        # trust its directions without re-deriving them.
        unsourced = [name for name in names if not self.provenance.get(name, "").strip()]
        if unsourced:
            raise ValueError(
                f"catalogue entries with no provenance: {sorted(unsourced)}. Record how "
                f"each guardrail's score direction and range were established."
            )

        # A guardrail cannot be both recommended and rejected. This has a real failure
        # mode behind it: an exclusion added while the entry stayed in the list would keep
        # offering a guardrail that someone had already found to be broken.
        contradictory = sorted(set(names) & set(self.exclusions))
        if contradictory:
            raise ValueError(
                f"catalogue both offers and excludes: {contradictory}. Remove it from one."
            )

        empty_reasons = sorted(
            name for name, reason in self.exclusions.items() if not reason.strip()
        )
        if empty_reasons:
            raise ValueError(
                f"excluded with no reason given: {empty_reasons}. Record the observation "
                f"that excluded it, so it is not re-tested and re-rejected later."
            )

    def definitions(self) -> tuple[GuardrailDefinition, ...]:
        """The offered guardrails, as copies.

        Copied so a caller adjusting a threshold on one cannot mutate the shared
        catalogue that every other caller reads.
        """
        return tuple(definition.model_copy(deep=True) for definition in self.entries)

    def exclusion_reason(self, guardrail_name: str) -> str | None:
        """Why a guardrail is not offered, or None if it is offered or simply unknown."""
        return self.exclusions.get(guardrail_name)


#: The catalogue you get when you supply none. Empty rather than illustrative: a filled-in
#: example would eventually be used as though it were verified.
EMPTY_CATALOGUE = GuardrailCatalogue()
