"""Whom to list for dates: strangers by expected mutual value, engaged partners first.

The engine pairs mutual interest before one-way interest, and within a stage it
takes the first eligible name of a set, not our favourite one. Every name on the
list is therefore roughly an equal draw for our next date, so the list should hold
only people who are likely to be good for us *and* to want us back.

For a stranger we know the trait they advertise and the value they seek on one
dimension. Everything else comes from a learned per-dimension distribution:
profiles and ideals are drawn from the same table, so traits seen on first dates
(plus our own profile and ideal, which are genuine samples) estimate both.

With M our total cost from their profile and T their total cost from ours,
	ours   = B - M - T / 2
	theirs = B - T - M / 2
M and T are independent sums of per-dimension costs. The ad pins one term of
each, so their distributions only depend on which dimensions the ad names; we
convolve the other four exactly, then read off
	P(ours > a and theirs > b)  and  E[ours * 1{both}]
with one pass over M and a cumulative table of T.
"""

import math

from models.player import Advertisement

VALUES = range(11)


class DateModel:
	"""Per-player cache of the cost distributions and the stranger scores."""

	def __init__(self, profile, ideal, monotonic: int, bonus: float) -> None:
		self.profile = tuple(profile)
		self.ideal = tuple(ideal)
		self.monotonic = monotonic
		self.dims = len(self.profile)
		self.bonus = float(bonus)
		self.n_obs = -1
		self.key: tuple = ()
		self.cache: dict[tuple, tuple[float, float]] = {}
		self.mine_rest: list[dict[int, float]] = []
		self.their_cdf: list[tuple[list[float], list[float]]] = []
		self.mean_mine: list[float] = []
		self.mean_theirs: list[float] = []

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	# ------------------------------------------------------------ learning

	def refresh(self, dim_values: list[list[int]], prior: float, min_growth: float) -> None:
		"""Rebuild the distributions when enough new observations have arrived."""
		n = sum(len(v) for v in dim_values)
		if self.n_obs >= 0 and n < self.n_obs * (1 + min_growth) and n - self.n_obs < 50:
			return
		self.n_obs = n
		self.cache.clear()
		pmfs = []
		for dim in range(self.dims):
			counts = [prior / 11] * 11
			for v in dim_values[dim]:
				counts[v] += 1
			counts[self.profile[dim]] += 1
			counts[self.ideal[dim]] += 1
			total = sum(counts)
			pmfs.append([c / total for c in counts])

		mine_dims, their_dims = [], []
		for dim, pmf in enumerate(pmfs):
			mine: dict[int, float] = {}
			theirs: dict[int, float] = {}
			for v, p in zip(VALUES, pmf, strict=True):
				c = self._gap(dim, self.ideal[dim], v)  # their trait v vs our ideal
				mine[c] = mine.get(c, 0.0) + p
				c = self._gap(dim, v, self.profile[dim])  # their ideal v vs our trait
				theirs[c] = theirs.get(c, 0.0) + p
			mine_dims.append(mine)
			their_dims.append(theirs)
		self.mean_mine = [sum(c * p for c, p in d.items()) for d in mine_dims]
		self.mean_theirs = [sum(c * p for c, p in d.items()) for d in their_dims]
		# The leave-one-out convolutions are only needed by joint() ('mutual' mode). They
		# grow with the number of dimensions (seconds at d=24), so build them on demand.
		self.dim_costs = (mine_dims, their_dims)
		self.mine_rest = None
		self.their_cdf = None

	def _build_rest(self) -> None:
		mine_dims, their_dims = self.dim_costs
		self.mine_rest = []
		self.their_cdf = []
		for skip in range(self.dims):
			self.mine_rest.append(_convolve([d for i, d in enumerate(mine_dims) if i != skip]))
			rest = _convolve([d for i, d in enumerate(their_dims) if i != skip])
			top = max(rest)
			cdf, partial = [0.0] * (top + 1), [0.0] * (top + 1)
			for c, p in rest.items():
				cdf[c] += p
				partial[c] += c * p
			for c in range(1, top + 1):
				cdf[c] += cdf[c - 1]
				partial[c] += partial[c - 1]
			self.their_cdf.append((cdf, partial))

	# ------------------------------------------------------------- scoring

	def expected(self, ad: Advertisement) -> float:
		"""E[our happiness] from the ad and the learned distributions."""
		od, sd = ad.own_dim, ad.seek_dim
		mine = sum(self.mean_mine) - self.mean_mine[od]
		mine += self._gap(od, self.ideal[od], ad.own_value)
		theirs = sum(self.mean_theirs) - self.mean_theirs[sd]
		theirs += self._gap(sd, ad.seek_value, self.profile[sd])
		return self.bonus - mine - theirs / 2

	def set_thresholds(self, ours: float, theirs: float) -> None:
		"""Bucket the acceptance thresholds so the score cache survives rounds."""
		step = max(1.0, self.bonus / 20)
		key = (round(ours / step), round(theirs / step))
		if key != self.key:
			self.key = key
			self.cache.clear()
		self.a, self.b = key[0] * step, key[1] * step

	def joint(self, ad: Advertisement) -> tuple[float, float]:
		"""(P(both clear their thresholds), E[our happiness * 1{both clear}])."""
		od, sd = ad.own_dim, ad.seek_dim
		m0 = self._gap(od, self.ideal[od], ad.own_value)
		t0 = self._gap(sd, ad.seek_value, self.profile[sd])
		key = (od, m0, sd, t0)
		hit = self.cache.get(key)
		if hit is not None:
			return hit
		if self.mine_rest is None:
			self._build_rest()
		cdf, partial = self.their_cdf[sd]
		top = len(cdf) - 1
		B, a, b = self.bonus, self.a, self.b
		prob = value = 0.0
		for m, pm in self.mine_rest[od].items():
			mm = m + m0
			# ours > a  <=>  T < 2 (B - a - mm);  theirs > b  <=>  T < B - b - mm / 2
			bound = min(2 * (B - a - mm), B - b - mm / 2)
			tmax = math.ceil(bound) - 1 - t0
			if tmax < 0:
				continue
			if tmax > top:
				tmax = top
			f, s = cdf[tmax], partial[tmax]
			prob += pm * f
			value += pm * ((B - mm) * f - 0.5 * (s + t0 * f))
		out = (prob, value)
		self.cache[key] = out
		return out


def _convolve(dists: list[dict[int, float]]) -> dict[int, float]:
	out = {0: 1.0}
	for d in dists:
		nxt: dict[int, float] = {}
		for c1, p1 in out.items():
			for c2, p2 in d.items():
				c = c1 + c2
				nxt[c] = nxt.get(c, 0.0) + p1 * p2
		out = nxt
	return out


# ---------------------------------------------------------------- listing


def choose(player, candidates, turn, params: dict) -> list[int]:
	"""The interest list in 'ev' or 'mutual' mode. See module docstring."""
	model: DateModel = player.date_model
	model.refresh(player.dim_values, params['PRIOR'], params['REFRESH_GROWTH'])

	limit = player.list_limit if player.list_limit is not None else len(candidates)
	bar = player._bar(turn.rounds_left)
	endgame = player._endgame(turn.rounds_left)
	relaxed = player._relaxed(turn.rounds_left, bar)
	floor = player._floor(turn.rounds_left)

	known = sorted(
		(
			c.index
			for c in candidates
			if c.index in player.scores
			and c.index not in player.blocked
			and player.paused_until.get(c.index, 0) < turn.round
		),
		key=lambda i: (
			player._unresponsive(i),
			# With RECIP_FIRST, people who showed interest back come first and people
			# who turned down another date last, so a cut keeps likely matches.
			-_recip(player, i, turn.round, params) if params['RECIP_FIRST'] else 0,
			-player.scores[i][0],
			i,
		),
	)
	willing_rounds = params['WILLING_ROUNDS']
	engaged = []
	strong = 0
	for i in known:
		score = player.scores[i][0]
		recent = turn.round - player.proposed_to_me.get(i, -100) <= willing_rounds
		# Someone who just proposed is worth listing even below the bar: they'd say yes.
		willing = params['LIST_WILLING'] and recent and score > floor
		if not willing and not (score >= bar and player._eligible(i, relaxed, endgame)):
			continue
		engaged.append(i)
		if score >= bar and (recent or player.their_again.get(i)):
			strong += 1
	top = params['KNOWN_TOP']
	if top:
		engaged = engaged[:top]

	fresh = [c for c in candidates if c.index not in player.reports]
	mode = params['MODE']
	samples = player.samples
	scarce = (
		params['SCARCE'] > 0
		and len(samples) >= params['SCARCE_MIN']
		and sum(1 for s in samples if s > 0) < params['SCARCE'] * len(samples)
	)
	if scarce:
		# Few people suit us: look for anyone both sides would be positive with.
		mode = 'scarce'
	if mode == 'ad2':
		fresh.sort(key=lambda c: (player._ad_cost(c.ad), c.index))
	elif mode == 'ev':
		fresh.sort(key=lambda c: (-model.expected(c.ad), c.index))
	else:
		if mode == 'scarce':
			ours = theirs = 0.0
			qw = 0.0
		else:
			left = max(0.0, (turn.rounds_left - 1) / max(1, player.rounds - 1))
			ours = max(bar, params['US_MIN_SHARE'] * model.bonus)
			theirs = params['THEM_SHARE'] * model.bonus * left
			qw = params['QUALITY_W'] / max(1.0, model.bonus)
		model.set_thresholds(ours, theirs)

		def key(c):
			p, v = model.joint(c.ad)
			return (-(p + qw * v), c.index)

		fresh.sort(key=key)

	share = params['FRESH_SHARE'] if turn.round <= params['TRIM_UNTIL'] * player.rounds else 1.0
	if scarce:
		share = 1.0
	elif strong:
		share = min(share, params['ENGAGED_FRESH_SHARE'])
	cap = max(params['MIN_FRESH'], math.ceil(share * limit))
	weak = []
	if endgame or not fresh:
		listed = set(engaged)
		weak = [
			i
			for i in known
			if i not in listed
			and player.scores[i][0] > floor
			and player._eligible(i, relaxed, endgame)
		]
		if top:
			weak = weak[: max(0, top - len(engaged))]
	chosen = engaged + [c.index for c in fresh[:cap]] + weak
	return chosen[:limit]


def _recip(player, i: int, round_number: int, params: dict) -> int:
	"""2 if they recently proposed, 1 if they asked for another date, -1 if they declined."""
	if round_number - player.proposed_to_me.get(i, -100) <= params['WILLING_ROUNDS']:
		return 2
	again = player.their_again.get(i)
	if again is None:
		return 0
	return 1 if again else -1
