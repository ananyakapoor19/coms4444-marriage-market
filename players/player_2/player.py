import math

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement

OBSERVE_FRACTION = 0.37
LIE_LAST = 0.3


class Player2(BasePlayer):
	"""Observe-then-commit player with a bar that drops as time runs out.

	Advertises and projects itself honestly, and takes what others show at face
	value. Its ad shows its best monotonic characteristic (highest value) and the
	monotonic characteristic it wants most.

	For the first 37% of rounds it only observes: it dates new people, scores
	each by the handout's happiness formula, and remembers the best score.

	After that, a person is good enough if their score is positive and at least
	the current bar. The bar starts at the best observed score and falls in a
	straight line to 0 by the last round, so we get less picky as time runs out
	instead of waiting for someone as good as the best early match.

	In the last 30% of rounds, when a date looks positive for us, we tell them
	their projected profile is exactly our ideal, so they think we would be fully
	happy and are more likely to propose back.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.cutoff = math.ceil(OBSERVE_FRACTION * self.rounds)
		self.scores: dict[int, float] = {}
		self.best_observed = 0.0
		self.rounds_left = self.rounds

	def advertise(self) -> Advertisement:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(showable, key=lambda i: self.ideal[i])
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def _bar(self) -> float:
		"""Best observed score right after observing, falling to 0 by the last round."""
		span = max(1, self.rounds - self.cutoff)
		return self.best_observed * max(0.0, (self.rounds_left - 1) / span)

	def _good_enough(self, score: float) -> bool:
		return score > 0 and score >= self._bar()

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.rounds_left = turn.rounds_left
		if turn.round <= self.cutoff:
			chosen = [c.index for c in candidates if c.index not in self.scores]
			self.rng.shuffle(chosen)
			return chosen[: self.list_limit]
		known = sorted(
			(c for c in candidates if self._good_enough(self.scores.get(c.index, 0.0))),
			key=lambda c: (-self.scores[c.index], c.index),
		)
		fresh = [c.index for c in candidates if c.index not in self.scores]
		self.rng.shuffle(fresh)
		return ([c.index for c in known] + fresh)[: self.list_limit]

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		late = date.rounds_left <= LIE_LAST * self.rounds
		if late and self.bonus - decrement(self.ideal, observed, self.monotonic) > 0:
			return observed
		return self.ideal

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		self.rounds_left = date.rounds_left
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		score = self.bonus - mine - theirs / 2
		self.scores[date.partner] = score

		if score <= 0:
			return Decision(incompatible=True)

		if date.round <= self.cutoff:
			self.best_observed = max(self.best_observed, score)
			return Decision()

		if not self._good_enough(score):
			return Decision()
		return Decision(again=True, propose=date.date_number >= 2)
