"""Group 4's player.

Dates the best-looking ads first, keeps a bar that starts high and falls
over the run, and proposes when the chance of a positive marriage clears
a bar that also falls over the run. Never proposes below its floor.
"""

from bisect import insort
from dataclasses import dataclass, field

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import MAX_VALUE, Profile, decrement

# The bar: a quantile of the scores we have seen, falling over the run.
EARLY_THRESHOLD_QUANTILE = 0.9
LATE_THRESHOLD_QUANTILE = 0.0
FLAT_THRESHOLD_SHARE = 0.1
MIN_SCORES_FOR_QUANTILE = 5
FALLBACK_THRESHOLD_FRACTION = 0.8
FULL_FALLBACK_ROUNDS = 20

# Small fields need a lower bar. With few partners, a 0.9 quantile and a
# 0.8B fallback are unreachable, so we marry almost nobody. At 100+ people
# this is exactly the old behaviour.
SMALL_FIELD_EARLY_QUANTILE = 0.5
SMALL_FIELD_FALLBACK_FRACTION = 0.4
FIELD_SIZE_REFERENCE = 100

# Polarized fields need a lower bar too. When most traits sit at 0 or 10,
# good matches are rare. We detect it from the profiles we see on dates.
POLARIZATION_MIN_VALUES = 20
POLARIZATION_THRESHOLD = 0.5

# Partners who lie get a penalty, need more dates, and a higher floor.
UNTRUTHFUL_PENALTY_MULTIPLIER = 1.2
UNTRUTHFUL_MIN_DATES = 3
UNTRUTHFUL_FLOOR_MULTIPLIER = 2.0

# Short games have no time for the 3-date check on liars. With 10 rounds
# or fewer, we can propose on date 2 to everyone.
SHORT_GAME_ROUNDS = 10

# Proposals: two failed proposals in a row means a 5-round cooldown.
FAILED_PROPOSALS_BEFORE_COOLDOWN = 2
ACCEPT_PROPOSAL_MARGIN_FRACTION = 0.1

# The hard floor below which we never propose.
SCORE_FLOOR_FRACTION = 0.05
SCORE_FLOOR_PER_DIM = 0.25

# In the last 3 rounds, marry our best partner so far if we can.
ENDGAME_ROUNDS = 3

# When a partner mirrors us, we have no real reading on that half, so we
# fall back to a quarter of the bonus as a prior.
MIRROR_PRIOR_FRACTION = 0.25


@dataclass
class DateRecord:
	"""One date with a partner, from our side."""

	round: int
	date_number: int
	expected_score: float
	we_proposed: bool = False


@dataclass
class PlayerProfile:
	"""Everything we know about one other player."""

	index: int
	advertisement: Advertisement | None = None
	projected_profile: Profile | None = None
	stated_ideal: Profile | None = None
	is_untruthful: bool = False
	expected_score: float | None = None

	last_informative_mine: float | None = None
	last_informative_theirs: float | None = None
	dating_history: list[DateRecord] = field(default_factory=list)
	failed_proposals_in_a_row: int = 0
	cooldown_last_round: int = 0
	accept_next_proposal_chance: bool = False
	# Willingness: how often they agreed to another date, and whether
	# they ever proposed to us. Used to judge if a proposal will land.
	repeat_count: int = 0
	date_count: int = 0
	proposed_to_us: bool = False

	@property
	def has_dated(self) -> bool:
		return self.expected_score is not None

	def is_in_cooldown(self, round_number: int) -> bool:
		return round_number <= self.cooldown_last_round


class Player4(BasePlayer):
	"""Player 4: patient bar, probability model, and a hard floor."""

	COOLDOWN_ROUNDS = 5

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.player_profiles: dict[int, PlayerProfile] = {}
		self.first_date_scores: list[float] = []
		self._mine_seen: list[float] = []
		self._theirs_seen: list[float] = []
		self.observed_values: list[int] = []
		self._last_projected: dict[int, Profile] = {}
		self._last_singles = self.people

	def _get_profile(self, index: int) -> PlayerProfile:
		if index not in self.player_profiles:
			self.player_profiles[index] = PlayerProfile(index=index)
		return self.player_profiles[index]

	# Scoring: how good a partner would be to marry.

	def _dim_penalty(self, dim: int, desired: int, actual: int) -> int:
		"""One-dimension decrement, same rule as models.profile.decrement."""
		if dim < self.monotonic and actual >= desired:
			return 0
		return (desired - actual) ** 2

	def _ad_penalty(self, advertisement: Advertisement) -> float:
		"""Lower is better: the penalty implied by their ad alone."""
		my_penalty = self._dim_penalty(
			advertisement.own_dim, self.ideal[advertisement.own_dim], advertisement.own_value
		)
		their_penalty = self._dim_penalty(
			advertisement.seek_dim, advertisement.seek_value, self.profile[advertisement.seek_dim]
		)
		return my_penalty + their_penalty / 2

	def _mirrored_value(self, seen: list[float]) -> float:
		"""Running mean of that half, or a prior if we have no data yet."""
		if seen:
			return sum(seen) / len(seen)
		return MIRROR_PRIOR_FRACTION * self.bonus

	def _expected_score(self, partner: PlayerProfile) -> float:
		"""Happiness if we married them, with mirrored halves replaced."""
		observed = partner.projected_profile
		stated_ideal = partner.stated_ideal

		mine_from_own = False
		if tuple(observed) == tuple(self.ideal):
			if partner.last_informative_mine is not None:
				mine = partner.last_informative_mine
				mine_from_own = True
			else:
				mine = self._mirrored_value(self._mine_seen)
			mine_mirrored = True
		else:
			mine = decrement(self.ideal, observed, self.monotonic)
			mine_mirrored = False

		theirs_from_own = False
		shown = self._last_projected.get(partner.index)
		if tuple(stated_ideal) == tuple(self.profile) or (
			shown is not None and tuple(stated_ideal) == tuple(shown)
		):
			if partner.last_informative_theirs is not None:
				theirs = partner.last_informative_theirs
				theirs_from_own = True
			else:
				theirs = self._mirrored_value(self._theirs_seen)
			theirs_mirrored = True
		else:
			theirs = decrement(stated_ideal, self.profile, self.monotonic)
			theirs_mirrored = False
		if not mine_mirrored:
			self._mine_seen.append(float(mine))
			partner.last_informative_mine = float(mine)
		if not theirs_mirrored:
			self._theirs_seen.append(float(theirs))
			partner.last_informative_theirs = float(theirs)
		total_penalty = mine + theirs / 2
		if partner.is_untruthful and not (mine_from_own and theirs_from_own):
			total_penalty *= UNTRUTHFUL_PENALTY_MULTIPLIER
		return self.bonus - total_penalty

	# The bar: what score is good enough to ask again or propose.

	def _ramp_progress(self, round_number: int) -> float:
		"""0 for the first 10% of rounds, 1 for the last 10%, linear in between."""
		progress = (round_number - 1) / self.rounds
		ramp_start = FLAT_THRESHOLD_SHARE
		ramp_end = 1 - FLAT_THRESHOLD_SHARE
		return min(max((progress - ramp_start) / (ramp_end - ramp_start), 0.0), 1.0)

	def _urgency(self, rounds_left: int) -> float:
		"""0 at the start, 1 when time or people run out."""
		time_frac = rounds_left / self.rounds
		people_frac = self._last_singles / self.people
		return min(max(1 - time_frac * people_frac, 0.0), 1.0)

	def _size_factor(self) -> float:
		"""1.0 at 100+ people, shrinking toward 0 in tiny fields."""
		return min(1.0, self.people / FIELD_SIZE_REFERENCE)

	def _polarization_factor(self) -> float:
		"""1.0 unless the traits we have seen are mostly at 0 or 10."""
		values = self.observed_values
		if len(values) < POLARIZATION_MIN_VALUES:
			return 1.0
		extreme = sum(1 for v in values if v == 0 or v == MAX_VALUE)
		share = extreme / len(values)
		if share <= POLARIZATION_THRESHOLD:
			return 1.0
		return 1.0 - (share - POLARIZATION_THRESHOLD)

	def _threshold_score(self, round_number: int) -> float:
		"""Minimum expected score to ask again or propose this round."""
		ramp_progress = self._ramp_progress(round_number)
		# The bar is set by whichever constraint is tighter: a small field
		# or a polarized field.
		factor = min(self._size_factor(), self._polarization_factor())
		early_quantile = (
			SMALL_FIELD_EARLY_QUANTILE
			+ (EARLY_THRESHOLD_QUANTILE - SMALL_FIELD_EARLY_QUANTILE) * factor
		)
		scores = self.first_date_scores
		if len(scores) < MIN_SCORES_FOR_QUANTILE:
			fallback_fraction = (
				SMALL_FIELD_FALLBACK_FRACTION
				+ (FALLBACK_THRESHOLD_FRACTION - SMALL_FIELD_FALLBACK_FRACTION) * factor
			)
			fraction = fallback_fraction * min(1.0, self.rounds / FULL_FALLBACK_ROUNDS)
			value = fraction * (1 - ramp_progress) * self.bonus
		else:
			quantile = early_quantile + (LATE_THRESHOLD_QUANTILE - early_quantile) * ramp_progress
			position = min(int(quantile * len(scores)), len(scores) - 1)
			value = scores[position]
		return value

	# Probability: how likely is a positive marriage with this partner.

	def _p_quality(self, partner: PlayerProfile) -> float:
		"""Chance this partner is a good match, from their score's rank."""
		if partner.expected_score is None or partner.expected_score <= 0:
			return 0.0
		scores = self.first_date_scores
		if not scores:
			return 0.5
		rank = sum(1 for s in scores if s <= partner.expected_score)
		p = (rank + 1) / (len(scores) + 2)
		if partner.is_untruthful:
			p *= 0.7
		return min(max(p, 0.0), 1.0)

	def _p_willing(self, partner: PlayerProfile) -> float:
		"""Chance they will accept a proposal, from how they acted."""
		good = partner.repeat_count + (1 if partner.proposed_to_us else 0)
		return (good + 1) / (partner.date_count + 2)

	def _aspiration_prob(self, rounds_left: int) -> float:
		"""The probability bar. Starts at 0.5, falls to near 0 at the end."""
		return 0.5 * (rounds_left / self.rounds)

	# Honesty: is this partner lying to us.

	def _check_truthfulness(
		self, partner: PlayerProfile, observed: Profile, stated_ideal: Profile
	) -> None:
		"""Flag them if this date contradicts their ad or their last date."""
		ad = partner.advertisement
		contradicts_ad = ad is not None and observed[ad.own_dim] != ad.own_value
		changed_since_last_date = partner.projected_profile is not None and (
			observed != partner.projected_profile or stated_ideal != partner.stated_ideal
		)
		mirrors_us = tuple(observed) == tuple(self.ideal) or tuple(stated_ideal) == tuple(
			self.profile
		)
		if contradicts_ad or changed_since_last_date or mirrors_us:
			partner.is_untruthful = True

	# The game, in order: advertise, pick dates, date, decide, learn.

	def advertise(self) -> Advertisement:
		"""Show a real strength and a real demand, so good matches find us."""
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own_dim = max(showable, key=lambda i: self.profile[i])
		seek_dim = max(range(self.dims), key=lambda i: abs(self.ideal[i] - MAX_VALUE // 2))
		return Advertisement(
			own_dim=own_dim,
			own_value=self.profile[own_dim],
			seek_dim=seek_dim,
			seek_value=self.ideal[seek_dim],
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self._last_singles = turn.singles
		threshold = self._threshold_score(turn.round)
		promising_dated: list[PlayerProfile] = []
		second_chance: list[PlayerProfile] = []
		never_dated: list[PlayerProfile] = []

		for candidate in candidates:
			partner = self._get_profile(candidate.index)
			partner.advertisement = candidate.ad
			if partner.is_in_cooldown(turn.round):
				continue
			if not partner.has_dated:
				never_dated.append(partner)
			elif partner.expected_score >= threshold or partner.accept_next_proposal_chance:
				promising_dated.append(partner)
			elif partner.expected_score <= 0 and candidate.dates < 2:
				second_chance.append(partner)

		promising_dated.sort(key=lambda partner: -partner.expected_score)
		second_chance.sort(key=lambda partner: -partner.expected_score)
		self.rng.shuffle(never_dated)
		never_dated.sort(key=lambda partner: self._ad_penalty(partner.advertisement))

		if turn.rounds_left <= ENDGAME_ROUNDS:
			best = self._best_so_far(turn.round)
			cand_idx = {c.index for c in candidates}
			if best is not None and best.index in cand_idx:
				ranked = [best.index]
			else:
				ranked = [partner.index for partner in promising_dated]
		else:
			ranked = [partner.index for partner in promising_dated + second_chance + never_dated]
			if turn.rounds_left <= 2 * ENDGAME_ROUNDS:
				best = self._best_so_far(turn.round)
				if (
					best is not None
					and best.index in {c.index for c in candidates}
					and best.index not in ranked
				):
					safe_at = len(promising_dated) + len(second_chance)
					ranked.insert(safe_at, best.index)
		return ranked[: self.list_limit]

	def _best_so_far(self, round_number: int) -> PlayerProfile | None:
		"""Best partner we have dated with a positive score, not in cooldown."""
		best: PlayerProfile | None = None
		for partner in self.player_profiles.values():
			if not partner.has_dated or partner.expected_score is None:
				continue
			if partner.expected_score <= 0:
				continue
			if partner.is_in_cooldown(round_number):
				continue
			if best is None or partner.expected_score > best.expected_score:
				best = partner
		return best

	def project(self, date: DateContext) -> Profile:
		"""Show our true profile."""
		result = self.profile
		self._last_projected[date.partner] = result
		return result

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		return self.ideal

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = self._get_profile(date.partner)
		partner.advertisement = date.partner_ad
		self._check_truthfulness(partner, observed, stated_ideal)
		partner.projected_profile = observed
		partner.stated_ideal = stated_ideal
		self.observed_values.extend(observed)
		partner.expected_score = self._expected_score(partner)

		date_record = DateRecord(
			round=date.round, date_number=date.date_number, expected_score=partner.expected_score
		)
		partner.dating_history.append(date_record)
		if date.date_number == 1:
			insort(self.first_date_scores, partner.expected_score)

		if partner.expected_score <= 0:
			if date.date_number >= 2:
				return Decision(incompatible=True)
			return Decision()

		threshold = self._threshold_score(date.round)
		floor = max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
		urgency = self._urgency(date.rounds_left)
		floor = floor * (1 - 0.5 * urgency)
		if partner.is_untruthful:
			floor = floor * UNTRUTHFUL_FLOOR_MULTIPLIER
		clears_threshold = partner.expected_score >= threshold
		if self.rounds <= SHORT_GAME_ROUNDS:
			can_propose = date.date_number >= 2
		else:
			can_propose = date.date_number >= (UNTRUTHFUL_MIN_DATES if partner.is_untruthful else 2)

		may_ask_again = not partner.is_in_cooldown(date.round)

		if partner.accept_next_proposal_chance:
			shortfall = threshold - partner.expected_score
			if shortfall > ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus:
				partner.accept_next_proposal_chance = False

		if date.rounds_left <= ENDGAME_ROUNDS:
			best = self._best_so_far(date.round)
			if best is not None and best.index == date.partner and can_propose:
				salvage_floor = 1.0 if not partner.is_untruthful else floor
				if partner.expected_score >= salvage_floor:
					date_record.we_proposed = True
					return Decision(again=may_ask_again, propose=True)

		# Propose when the chance of a positive marriage clears the bar.
		# The bar falls as rounds run out, so we get less picky late.
		p_marry = self._p_quality(partner) * self._p_willing(partner)
		bar = self._aspiration_prob(date.rounds_left)
		if (
			can_propose
			and partner.expected_score >= floor
			and (p_marry >= bar or partner.accept_next_proposal_chance)
		):
			date_record.we_proposed = True
			return Decision(again=may_ask_again, propose=True)
		if clears_threshold:
			return Decision(again=True)

		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		partner = self._get_profile(outcome.partner)
		if outcome.married or not partner.dating_history:
			return
		date_record = partner.dating_history[-1]

		# Track willingness for the probability model.
		partner.date_count += 1
		if outcome.again:
			partner.repeat_count += 1
		if outcome.partner_proposed:
			partner.proposed_to_us = True

		if date_record.we_proposed:
			partner.failed_proposals_in_a_row += 1
			partner.accept_next_proposal_chance = False
			if partner.failed_proposals_in_a_row >= FAILED_PROPOSALS_BEFORE_COOLDOWN:
				partner.cooldown_last_round = outcome.round + self.COOLDOWN_ROUNDS
				partner.failed_proposals_in_a_row = 0

		if outcome.partner_proposed and not date_record.we_proposed:
			shortfall = self._threshold_score(outcome.round) - date_record.expected_score
			floor = max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
			floor = floor * (1 - 0.5 * self._urgency(self.rounds - outcome.round))
			if partner.is_untruthful:
				floor = floor * UNTRUTHFUL_FLOOR_MULTIPLIER
			if (
				date_record.expected_score >= floor
				and shortfall <= ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus
			):
				partner.accept_next_proposal_chance = True
