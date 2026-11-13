"""Group 9: truthful search, verified coordination, and individual happiness floors.

Screen public ads before coordinating. Verify reports through actual dates,
maximize positive marriages, and accept repairs only when nobody's estimate falls.
Use independent search when the common dating protocol cannot be verified.
"""

from collections import deque
from functools import cache
from itertools import combinations
from math import exp

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
from models.profile import VALUES, Profile, decrement, happiness


class Player9(BasePlayer):
	"""One policy with coordinated matching and an independent-search fallback."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.date_scores: dict[int, float] = {}
		self.proposed_to_us: set[int] = set()
		self.reports: dict[int, tuple[Profile, Profile] | None] = {}
		self.singles = self.people
		self.target_chosen = False
		self.target: int | None = None

		small = (
			8 <= self.people <= 22
			and self.monotonic >= 2
			and self.rounds >= 2 * self.people + 6
			and self.bonus > 0
		)
		self.large_eligible = (
			self.people == 50
			and self.dims in (5, 10)
			and self.monotonic == 2
			and self.bonus in (100, 150)
			and self.rounds in (100, 250)
		)
		self.coordinating: bool | None = None if small else False
		self.match_round = (
			self.people if self.large_eligible else min(2 * self.people, self.rounds // 2)
		)
		self.large_checked = False
		self.large_active = False
		self.roster: list[int] = []
		self.public_ad = _canonical_ad(self.profile, self.ideal, self.monotonic, small)

		# Uniform 0–10 expectations rank the information revealed by each ad.
		self.unknown_trait_cost = tuple(
			sum(self._cost(dim, want, value) for value in VALUES) / len(VALUES)
			for dim, want in enumerate(self.ideal)
		)
		self.unknown_preference_cost = tuple(
			sum(self._cost(dim, value, have) for value in VALUES) / len(VALUES)
			for dim, have in enumerate(self.profile)
		)

	def advertise(self) -> Advertisement:
		return self.public_ad

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.singles = turn.singles
		if not self.large_eligible:
			return self._independent_dates(candidates, turn)

		# Every exploration day must add its scheduled partner, including day 49.
		if (
			self.large_active
			and turn.round <= self.people
			and (len(candidates) < self.people - 1 or len(self.reports) < turn.round - 1)
		):
			self._stop_coordination()
		if not self.large_checked:
			self.large_checked = True
			self.roster = sorted([self.index] + [c.index for c in candidates])
			self.large_active = (
				len(candidates) == self.people - 1
				and all(_feasible_ad(c.ad, self.dims, self.monotonic) for c in candidates)
				and _market_score(
					[self.public_ad] + [c.ad for c in candidates], self.dims, _MARKET_WEIGHTS
				)
				>= 0.5
			)
			self.coordinating = self.large_active
		if not self.large_active:
			return self._independent_dates(candidates, turn)
		if turn.round < self.people:
			return [self._scheduled(turn.round)]

		available = {c.index for c in candidates}
		if not self.target_chosen:
			self.target = self._large_target(available)
			self.target_chosen = True
		if self.target in available:
			return [self.target]
		return self._independent_dates(candidates, turn)

	def _stop_coordination(self) -> None:
		self.large_active = False
		self.coordinating = False

	def _scheduled(self, round_number: int) -> int:
		"""Circle-method schedule: meet each other person once in 49 rounds."""
		rest = self.roster[1:]
		shift = (round_number - 1) % len(rest)
		if shift:
			rest = rest[-shift:] + rest[:-shift]
		ring = [self.roster[0]] + rest
		return ring[-1 - ring.index(self.index)]

	def _cost(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		return (max(0, gap) if dim < self.monotonic else gap) ** 2

	def _threshold(self, rounds_left: int) -> float:
		time_left = (rounds_left - 1) / max(1, self.rounds - 1)
		# Repeat dates use the last count observed while making a shortlist.
		options_left = (self.singles - 2) / max(1, self.people - 2)
		return 0.4 * self.bonus * min(time_left, options_left)

	def _ad_score(self, ad: Advertisement) -> float:
		# Rank by how much the advertised values improve on a fully unknown partner.
		trait_gain = self.unknown_trait_cost[ad.own_dim] - self._cost(
			ad.own_dim, self.ideal[ad.own_dim], ad.own_value
		)
		preference_gain = self.unknown_preference_cost[ad.seek_dim] - self._cost(
			ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim]
		)
		return trait_gain + preference_gain / 2

	def _independent_dates(
		self, candidates: tuple[Candidate, ...], turn: RoundContext
	) -> list[int]:
		if self.coordinating is None:
			self.coordinating = len(candidates) == self.people - 1 and all(
				candidate.ad.own_dim == 1 for candidate in candidates
			)
		if self.coordinating:
			if turn.round < self.match_round:
				fresh = sorted(
					(c for c in candidates if c.dates == 0),
					key=lambda c: self._ad_score(c.ad),
					reverse=True,
				)
				width = 6 if self.list_limit is None else min(6, self.list_limit)
				return [c.index for c in fresh[:width]]
			if not self.target_chosen:
				self.target = self._small_target(candidates)
				self.target_chosen = True
			if self.target is not None and any(c.index == self.target for c in candidates):
				return [self.target]

		if self.bonus <= 0:
			return []

		threshold = self._threshold(turn.rounds_left)
		known = []
		fresh = []
		for candidate in candidates:
			index = candidate.index
			score = self.date_scores.get(index)
			if score is None:
				if turn.rounds_left > 1:
					fresh.append((self._ad_score(candidate.ad), self.rng.random(), index))
				continue
			if score > (0 if index in self.proposed_to_us else threshold):
				known.append((score, index))

		# Ranking determines who makes the shortlist. The engine treats it as a set.
		known.sort(reverse=True)
		fresh.sort(reverse=True)
		width = 3 if self.list_limit is None else min(3, self.list_limit)
		return [entry[-1] for entry in (known + fresh)[:width]]

	def _positive_utilities(self, people):
		"""Build the same mutually positive graph for either matching mode."""
		ids = sorted(people)
		utilities = {}
		for a, b in combinations(range(len(ids)), 2):
			profile_a, ideal_a = people[ids[a]]
			profile_b, ideal_b = people[ids[b]]
			cost_a = decrement(ideal_a, profile_b, self.monotonic)
			cost_b = decrement(ideal_b, profile_a, self.monotonic)
			happy_a = self.bonus - cost_a - cost_b / 2
			happy_b = self.bonus - cost_b - cost_a / 2
			if happy_a > 0 and happy_b > 0:
				utilities[a, b] = happy_a, happy_b
		return ids, utilities

	def _large_target(self, available: set[int]) -> int | None:
		people = {self.index: (self.profile, self.ideal)}
		people.update(
			{
				i: report
				for i, report in self.reports.items()
				if i in available and report is not None
			}
		)
		ids, utilities = self._positive_utilities(people)
		mates = _quality_matching(len(ids), utilities, self.bonus)
		partner = mates[ids.index(self.index)]
		return ids[partner] if partner >= 0 else None

	def _small_target(self, candidates: tuple[Candidate, ...]) -> int | None:
		"""Solve small markets exactly, prioritizing pair count before happiness."""
		people = {self.index: (self.profile, self.ideal)}
		for candidate in candidates:
			report = self.reports.get(candidate.index)
			if report is not None:
				people[candidate.index] = report
		indices, utilities = self._positive_utilities(people)
		weights = {edge: sum(scores) for edge, scores in utilities.items()}

		choices: dict[int, int | None] = {}

		@cache
		def solve(mask: int) -> tuple[int, float]:
			if not mask:
				return 0, 0.0
			bit = mask & -mask
			first = bit.bit_length() - 1
			rest = mask ^ bit
			best, partner = solve(rest), None
			remaining = rest
			while remaining:
				bit = remaining & -remaining
				remaining ^= bit
				second = bit.bit_length() - 1
				weight = weights.get((first, second))
				if weight is not None:
					count, total = solve(rest ^ bit)
					if (count + 1, total + weight) > best:
						best, partner = (count + 1, total + weight), second
			choices[mask] = partner
			return best

		mask = (1 << len(indices)) - 1
		solve(mask)
		solve.cache_clear()
		while mask:
			bit = mask & -mask
			first = bit.bit_length() - 1
			second = choices[mask]
			if indices[first] == self.index:
				return None if second is None else indices[second]
			mask ^= bit
			if second is not None:
				if indices[second] == self.index:
					return indices[first]
				mask ^= 1 << second
		return None

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		if (
			self.large_active
			and date.round < self.people
			and (
				date.partner != self._scheduled(date.round)
				or _canonical_ad(observed, stated_ideal, self.monotonic) != date.partner_ad
			)
		):
			self._stop_coordination()
		report = observed, stated_ideal
		if self.reports.setdefault(date.partner, report) != report:
			self.reports[date.partner] = None
		score = happiness(
			self.bonus, self.profile, self.ideal, observed, stated_ideal, self.monotonic
		)
		self.date_scores[date.partner] = score

		has_next_round = date.rounds_left > 1
		if self.coordinating and date.round < self.match_round:
			return Decision()
		if date.partner == self.target:
			partner_score = happiness(
				self.bonus,
				observed,
				stated_ideal,
				self.profile,
				self.ideal,
				self.monotonic,
			)
			if score > 0 and partner_score > 0:
				return Decision(
					again=has_next_round and date.date_number <= 2,
					propose=date.date_number >= 2,
				)
			self.target = None
		if date.date_number == 1:
			return Decision(again=has_next_round and score > 0)

		threshold = (
			0.0 if date.partner in self.proposed_to_us else self._threshold(date.rounds_left)
		)
		propose = score > threshold
		# A proposal on date two can get one immediate follow-up.
		again = has_next_round and date.date_number == 2 and propose
		return Decision(again=again, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_us.add(outcome.partner)
		elif outcome.partner == self.target and outcome.date_number >= 2:
			self.target = None


def _pair_matching(count, weights):
	"""Maximize pair count with blossom, then apply bounded two-pair improvements."""
	neighbours = [[] for _ in range(count)]
	for a, b in weights:
		neighbours[a].append(b)
		neighbours[b].append(a)
	for a in range(count):
		neighbours[a].sort(key=lambda b: (-weights[min(a, b), max(a, b)], b))
	mate = [-1] * count
	for a, b in sorted(weights, key=lambda edge: (-weights[edge], edge)):
		if mate[a] < 0 and mate[b] < 0:
			mate[a] = b
			mate[b] = a

	def augment(root):
		parent = [-1] * count
		base = list(range(count))
		used = [False] * count

		def lca(a, b):
			path = [False] * count
			while True:
				a = base[a]
				path[a] = True
				if mate[a] < 0:
					break
				a = parent[mate[a]]
			while not path[base[b]]:
				b = parent[mate[base[b]]]
			return base[b]

		def mark(a, b, common, blossom):
			while base[a] != common:
				blossom[base[a]] = blossom[base[mate[a]]] = True
				parent[a] = b
				b = mate[a]
				a = parent[mate[a]]

		queue = deque([root])
		used[root] = True
		while queue:
			a = queue.popleft()
			for b in neighbours[a]:
				if base[a] == base[b] or mate[a] == b:
					continue
				if b == root or (mate[b] >= 0 and parent[mate[b]] >= 0):
					common = lca(a, b)
					blossom = [False] * count
					mark(a, b, common, blossom)
					mark(b, a, common, blossom)
					for i in range(count):
						if blossom[base[i]]:
							base[i] = common
							if not used[i]:
								used[i] = True
								queue.append(i)
				elif parent[b] < 0:
					parent[b] = a
					if mate[b] < 0:
						while b >= 0:
							a = parent[b]
							previous = mate[a] if a >= 0 else -1
							mate[b] = a
							if a >= 0:
								mate[a] = b
							b = previous
						return True
					used[mate[b]] = True
					queue.append(mate[b])
		return False

	for a in range(count):
		if mate[a] < 0:
			augment(a)

	# Keep cardinality fixed. These local swaps improve, but do not globally
	# maximize, happiness. A fixed iteration bound limits callback time.
	def weight(a, b):
		return weights.get((min(a, b), max(a, b)))

	for _ in range(count):
		pairs = [(a, b) for a, b in enumerate(mate) if a < b]
		best = 0
		change = None
		for i, (a, b) in enumerate(pairs):
			old = weight(a, b)
			for c in range(count):
				if mate[c] < 0:
					for u, v in [(a, b), (b, a)]:
						w = weight(u, c)
						if w is not None and w - old > best:
							best = w - old
							change = [(u, c), (v, -1)]
			for c, d in pairs[i + 1 :]:
				old_both = old + weight(c, d)
				for u, v in [(c, d), (d, c)]:
					x, y = weight(a, u), weight(b, v)
					if x is not None and y is not None and x + y - old_both > best:
						best = x + y - old_both
						change = [(a, u), (b, v)]
		if change is None:
			break
		for a, b in change:
			mate[a] = b
			if b >= 0:
				mate[b] = a
	return mate


def _match_graph(count, weights):
	"""Refine the matching with up to six three-pair improvements."""
	mate = _pair_matching(count, weights)
	w = [[None] * count for _ in range(count)]
	for (a, b), value in weights.items():
		w[a][b] = w[b][a] = value
	# These eight rewires replace all three original pairs in a six-vertex cycle.
	patterns = [
		((0, 2), (1, 4), (3, 5)),
		((0, 2), (1, 5), (3, 4)),
		((0, 3), (1, 4), (2, 5)),
		((0, 3), (1, 5), (2, 4)),
		((0, 4), (1, 2), (3, 5)),
		((0, 4), (1, 3), (2, 5)),
		((0, 5), (1, 2), (3, 4)),
		((0, 5), (1, 3), (2, 4)),
	]
	for _ in range(6):
		pairs = [(a, b) for a, b in enumerate(mate) if a < b]
		best = 0
		change = None
		for first, second, third in combinations(pairs, 3):
			nodes = first + second + third
			old = sum(w[a][b] for a, b in [first, second, third])
			for pattern in patterns:
				total = 0
				for a, b in pattern:
					value = w[nodes[a]][nodes[b]]
					if value is None:
						break
					total += value
				else:
					if total - old > best:
						best = total - old
						change = [(nodes[a], nodes[b]) for a, b in pattern]
		if change is None:
			break
		for a, b in change:
			mate[a] = b
			mate[b] = a
	return mate


# Frozen logistic coefficients for the public-ad coordination screen.
_MARKET_WEIGHTS = (
	-29.306378396467693,
	-1.2014740269010933,
	4.547772360467847,
	-2.7702571654514565,
	39.96725171319136,
	7.678886121389771,
	-27.719066542285518,
	15.944131223605613,
	-1.6743002752943716,
	-3.238163971811184,
	-3.732454735375161,
	-4.130395850173559,
	-0.41737651029444667,
	3.4936361267065386,
	-6.741819034418722,
	2.1581212880194314,
	1.4873817730274235,
	0.8245426752370812,
	4.73371725609194,
	-0.7814558276613365,
	-0.6263518965345065,
	-4.328337393289825,
	1.558080227838293,
	0.25591696762699856,
	10.711023990123651,
	3.327527326018112,
	-0.9800429990877282,
	2.1613588655790674,
	-9.091259410634907,
	0.1461956375466847,
	-3.7604632117205146,
)


def _market_features(ads, dims, monotonic=2):
	n = len(ads)
	return (
		[
			1.0,
			dims / 10,
			sum(a.own_dim >= monotonic for a in ads) / n,
			sum(a.seek_dim < monotonic for a in ads) / n,
			sum(a.own_value for a in ads) / (10 * n),
			sum(a.seek_value for a in ads) / (10 * n),
			sum(abs(a.own_value - 5) for a in ads) / (5 * n),
			sum(abs(a.seek_value - 5) for a in ads) / (5 * n),
			sum(a.own_value == 5 for a in ads) / n,
			sum(a.own_value == 10 for a in ads) / n,
			sum(a.seek_value in (0, 10) for a in ads) / n,
		]
		+ [sum(a.own_dim == i for a in ads) / n for i in range(10)]
		+ [sum(a.seek_dim == i for a in ads) / n for i in range(10)]
	)


def _market_score(ads, dims, weights):
	z = sum(a * b for a, b in zip(_market_features(ads, dims), weights, strict=True))
	return 1 / (1 + exp(-max(-35, min(35, z))))


def _canonical_ad(profile, ideal, monotonic, small_convention=False):
	def cost(i, want, have):
		return (max(0, want - have) if i < monotonic else want - have) ** 2

	own = min(range(len(profile)), key=lambda i: sum(cost(i, v, profile[i]) for v in VALUES))
	if small_convention:
		own = 1
	seek = max(range(len(profile)), key=lambda i: sum(cost(i, ideal[i], v) for v in VALUES))
	return Advertisement(own, profile[own], seek, ideal[seek])


def _feasible_ad(ad, dims, monotonic):
	"""Reject ads that our truthful argmin/argmax rule could never produce."""

	def cost(dim, want, have):
		return (max(0, want - have) if dim < monotonic else want - have) ** 2

	own = sum(cost(ad.own_dim, v, ad.own_value) for v in VALUES)
	# Every dimension has maximum undisclosed-preference cost385. Earlier
	# dimensions win ties, so later dimensions cannot win at this maximum.
	if ad.own_dim > 0 and own == 385:
		return False
	seek = sum(cost(ad.seek_dim, ad.seek_value, v) for v in VALUES)
	for dim in range(dims):
		minimum = 0 if dim < monotonic else 110
		if dim != ad.seek_dim and (minimum > seek or (dim < ad.seek_dim and minimum == seek)):
			return False
	return True


def _matching_values(mates, utilities):
	"""Return one happiness estimate per person; singles receive zero."""
	values = [0.0] * len(mates)
	for a, b in enumerate(mates):
		if a < b:
			values[a], values[b] = utilities[a, b]
	return values


def _matching_quality(mates, utilities, bonus):
	"""Measure coverage, welfare, weaker matches and the high-score tail."""
	values = _matching_values(mates, utilities)
	positive = sorted(value for value in values if value > 0)
	quarter = max(1, (len(positive) + 3) // 4)
	return {
		'count': len(positive),
		'total': sum(values),
		'lower': sum(positive[:quarter]),
		'peak': max(values, default=0),
		'near995': sum(value >= 0.995 * bonus for value in values),
		'near95': sum(value >= 0.95 * bonus for value in values),
		'near90': sum(value >= 0.90 * bonus for value in values),
	}


def _quality_matching(count, utilities, bonus):
	"""Improve exceptional scores without lowering any person's estimate."""
	weights = {edge: sum(scores) for edge, scores in utilities.items()}
	original = _match_graph(count, weights)
	baseline = _matching_quality(original, utilities, bonus)
	floors = _matching_values(original, utilities)
	best, best_quality = original, baseline

	def rank(quality):
		return tuple(
			quality[key] for key in ('peak', 'near995', 'near95', 'near90', 'total', 'lower')
		)

	def consider(mates):
		nonlocal best, best_quality
		# Personal floors also preserve total, lower-quarter, peak and tail scores.
		safe = all(
			value >= floor
			for value, floor in zip(_matching_values(mates, utilities), floors, strict=True)
		)
		if not safe:
			return
		quality = _matching_quality(mates, utilities, bonus)
		if quality['count'] == baseline['count'] and rank(quality) > rank(best_quality):
			best, best_quality = mates, quality

	# Try at most six outstanding pairs whose own members meet their floors.
	# Blossom plus swaps must also repair the rest without harming anyone.
	edges = sorted(utilities, key=lambda edge: (-max(utilities[edge]), -sum(utilities[edge]), edge))
	for a, b in edges[:6]:
		if original[a] == b or any(
			value < floors[i] for i, value in zip((a, b), utilities[a, b], strict=True)
		):
			continue
		ids = [i for i in range(count) if i not in (a, b)]
		back = {i: j for j, i in enumerate(ids)}
		residual = {
			(back[u], back[v]): weight
			for (u, v), weight in weights.items()
			if u in back and v in back
		}
		repaired = _match_graph(len(ids), residual)
		mates = [-1] * count
		mates[a], mates[b] = b, a
		for i, j in enumerate(repaired):
			if j >= 0:
				mates[ids[i]] = ids[j]
		consider(mates)

	# A convex reward finds another candidate with stronger individual scores.
	# The guard evaluates original linear happiness before accepting it.
	consider(
		_match_graph(
			count,
			{
				edge: (first * first + second * second) / bonus
				for edge, (first, second) in utilities.items()
			},
		)
	)
	return best
