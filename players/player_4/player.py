from models.player import (
	Candidate,
	DateContext,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement


class Player4(BasePlayer):
	"""Group 4's basic player.

	Honest throughout: default advertisement, true projected profile, true
	described ideal. After each date it estimates its happiness with the
	handout's formula, taking what it observed and what it was told at face
	value. A positive estimate means ask again and, from the second date on,
	propose; anything else marks the partner incompatible.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.estimates: dict[int, float] = {}

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		# Promising past dates first, then everyone we have not met yet.
		liked = sorted(
			(c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0),
			key=lambda i: -self.estimates[i],
		)
		unseen = [c.index for c in candidates if c.index not in self.estimates]
		self.rng.shuffle(unseen)
		return (liked + unseen)[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		estimate = self.bonus - mine - theirs / 2
		self.estimates[date.partner] = estimate

		if estimate <= 0:
			return Decision(incompatible=True)
		return Decision(again=True, propose=date.date_number >= 2)
