"""A dynamic-programming reservation value for Player1 (BAR_MODE = 'dp').

Utility puts the positive rate first: U(h) = h + lam * B * [h > 0], and staying
single is worth 0. With k meaningful chances left, the value of carrying on is

    V_0 = value of the partners we already know (a fallback, see ``pool_value``)
    V_k = V_{k-1} + q * E[(U(X) - V_{k-1})+]

where X is a recent first-date score and q the measured chance that a partner we
want wants us back. We accept a partner (again, then propose from date 2) when
U(score) >= V_k, so the bar falls continuously as rounds, strangers or singles
run out, and there is no separate endgame, scarce or big-market mode.

A known partner is worth another date when p_commit * (U - V_{k-1}) is at least
what a fresh date adds, V_k - V_{k-1}; people who keep dating us without
committing lose that test as their p_commit falls.
"""

import math
from bisect import bisect_right
from collections import deque


class DPBar:
	def __init__(
		self,
		bonus: float,
		lam: float,
		window: int,
		ramp_share: float = 0.0,
		q1_prior: float = 0.5,
		q2_prior: float = 0.7,
		prior_weight: float = 4.0,
		fresh_window: int = 10,
		decay_window: int = 10,
		rho_prior: float = 0.0,
		rho_weight: float = 0.0,
		mutual_cap: int = 1000,
	) -> None:
		self.mutual_cap = mutual_cap
		self.bonus = bonus
		self.rho_prior, self.rho_weight = rho_prior, rho_weight
		self.lam_b = lam * bonus
		# The positivity bonus ramps up over the first ramp_share * B of score, since a
		# barely positive score is often a hidden lie away from a bad marriage.
		self.ramp = ramp_share * bonus
		self.samples: deque[float] = deque(maxlen=window if window else None)
		self._sorted: list[float] | None = None
		self._suffix: list[float] = []
		self._prior: list[float] = []
		self._prior_suffix: list[float] = []
		self.prior_weight_n = 0.0
		# Stage 1: we asked for another date after a first date; did they too?
		self.q1_prior, self.q1_n, self.q1_s = q1_prior, 0, 0
		# Stage 2: we proposed to them; did they ever propose to us?
		self.q2_prior, self.q2_n, self.q2_s = q2_prior, 0, 0
		self.prior_weight = prior_weight
		self.q2_open: set[int] = set()
		self.q2_done: set[int] = set()
		# Rounds in which we had a first date, and (round, singles) as seen.
		self.fresh_window = fresh_window
		self.fresh_rounds: deque[int] = deque()
		self.singles: deque[tuple[int, int]] = deque(maxlen=decay_window + 1)
		# Per-partner evidence about commitment.
		self.proposals: dict[int, int] = {}
		self.mutual: dict[int, int] = {}
		self.ignored: dict[int, int] = {}
		self.snubbed: dict[int, int] = {}
		self.noshow: dict[int, int] = {}
		self.asked: dict[int, bool] = {}  # partner -> we said again on our latest date
		self.proposed: set[int] = set()  # partners we proposed to on our latest date

	# ------------------------------------------------------------ evidence

	def utility(self, score: float) -> float:
		if score <= 0:
			return score
		if score < self.ramp:
			return score + self.lam_b * score / self.ramp
		return score + self.lam_b

	def add_sample(self, score: float, rnd: int) -> None:
		self.samples.append(score)
		self._sorted = None
		if not self.fresh_rounds or self.fresh_rounds[-1] != rnd:
			self.fresh_rounds.append(rnd)

	def note_singles(self, rnd: int, singles: int) -> None:
		if not self.singles or self.singles[-1][0] != rnd:
			self.singles.append((rnd, max(1, singles)))

	def note_decision(self, partner: int, again: bool, propose: bool) -> None:
		self.asked[partner] = again
		if propose:
			self.proposed.add(partner)
			if partner not in self.q2_done:
				self.q2_open.add(partner)
		else:
			self.proposed.discard(partner)

	def note_outcome(self, partner: int, date_number: int, proposed: bool, mutual: bool) -> None:
		we_asked = self.asked.pop(partner, False)
		we_proposed = partner in self.proposed
		self.proposed.discard(partner)
		if date_number == 1 and we_asked:
			self.q1_n += 1
			self.q1_s += mutual or proposed
		if mutual:
			self.mutual[partner] = self.mutual.get(partner, 0) + 1
		elif we_asked and not proposed:
			self.snubbed[partner] = self.snubbed.get(partner, 0) + 1
		if proposed:
			self.proposals[partner] = self.proposals.get(partner, 0) + 1
			if partner in self.q2_open:
				self._resolve(partner, True)
		elif we_proposed:
			self.ignored[partner] = self.ignored.get(partner, 0) + 1
			if partner in self.q2_open and self.ignored[partner] >= 2:
				self._resolve(partner, False)

	def note_noshow(self, partner: int) -> None:
		self.noshow[partner] = self.noshow.get(partner, 0) + 1

	def _resolve(self, partner: int, success: bool) -> None:
		self.q2_open.discard(partner)
		self.q2_done.add(partner)
		self.q2_n += 1
		self.q2_s += success

	# ------------------------------------------------------------ estimates

	def _beta(self, s: float, n: float, prior: float) -> float:
		return (s + self.prior_weight * prior) / (n + self.prior_weight)

	def q1(self) -> float:
		return self._beta(self.q1_s, self.q1_n, self.q1_prior)

	def q2(self) -> float:
		return self._beta(self.q2_s, self.q2_n, self.q2_prior)

	def q(self) -> float:
		return self.q1() * self.q2()

	def p_commit(self, partner: int) -> float:
		"""Chance this known partner marries us if we keep at it."""
		props = self.proposals.get(partner, 0)
		mutual = self.mutual.get(partner, 0)
		base = self.q2() if props or mutual else self.q()
		a = 2.0 * base + 1.5 * props + 0.5 * min(mutual, self.mutual_cap)
		b = 2.0 * (1 - base)
		b += self.ignored.get(partner, 0) + 0.5 * self.snubbed.get(partner, 0)
		b += 0.1 * self.noshow.get(partner, 0)
		return a / (a + b)

	def fresh_rate(self, rnd: int) -> float:
		"""First dates per round lately, smoothed toward one."""
		start = rnd - self.fresh_window
		while self.fresh_rounds and self.fresh_rounds[0] < start:
			self.fresh_rounds.popleft()
		seen = sum(1 for r in self.fresh_rounds if start <= r < rnd)
		span = min(self.fresh_window, max(0, rnd - 1))
		return (seen + 1.0) / (span + 1.0)

	def decay(self) -> float:
		"""Per-round hazard of the market's singles disappearing.

		Measured over the last few rounds, and smoothed toward rho_prior (worth
		rho_weight rounds) so that the quiet first rounds do not read as "nobody
		will ever marry".
		"""
		measured, span = 0.0, 0
		if len(self.singles) >= 2:
			(r0, s0), (r1, s1) = self.singles[0], self.singles[-1]
			if r1 > r0:
				measured, span = max(0.0, math.log(s0 / s1) / (r1 - r0)), r1 - r0
		if not self.rho_weight:
			return measured
		return (measured * span + self.rho_prior * self.rho_weight) / (span + self.rho_weight)

	def chances(self, rnd: int, rounds_left: int, unmet: int) -> float:
		"""Meaningful first dates still to come after this round's."""
		future = max(0, rounds_left - 1)
		if not future:
			return 0.0
		rho = self.decay()
		f = self.fresh_rate(rnd)
		if rho < 1e-9:
			horizon = float(future)
		else:
			e = math.exp(-rho)
			horizon = e * (1 - e**future) / (1 - e)
		return max(0.0, min(f * horizon, float(unmet)))

	# ----------------------------------------------------------------- DP

	def _prepare(self) -> None:
		if self._sorted is None:
			utils = sorted(self.utility(s) for s in self.samples)
			suffix = [0.0] * (len(utils) + 1)
			for i in range(len(utils) - 1, -1, -1):
				suffix[i] = suffix[i + 1] + utils[i]
			self._sorted, self._suffix = utils, suffix

	def set_prior(self, scores: list[float], weight: float) -> None:
		"""Synthetic first-date scores, worth ``weight`` samples, fading as real ones arrive."""
		utils = sorted(self.utility(s) for s in scores)
		suffix = [0.0] * (len(utils) + 1)
		for i in range(len(utils) - 1, -1, -1):
			suffix[i] = suffix[i + 1] + utils[i]
		self._prior, self._prior_suffix, self.prior_weight_n = utils, suffix, weight

	@staticmethod
	def _mean_excess(utils: list[float], suffix: list[float], v: float) -> float:
		n = len(utils)
		if not n:
			return 0.0
		i = bisect_right(utils, v)
		return (suffix[i] - v * (n - i)) / n

	def excess(self, v: float) -> float:
		"""E[(U(X) - v)+] over the sample window (plus the prior, while it lasts)."""
		self._prepare()
		n = len(self._sorted)
		real = self._mean_excess(self._sorted, self._suffix, v)
		m = max(0.0, self.prior_weight_n - n) if self._prior else 0.0
		if not m:
			return real
		prior = self._mean_excess(self._prior, self._prior_suffix, v)
		return (n * real + m * prior) / (n + m)

	def values(self, k: float, v0: float) -> tuple[float, float]:
		"""(V_k, V_{k-1}) for a possibly fractional k, starting from V_0 = v0."""
		q = self.q()
		v_prev = v = v0
		whole = int(k)
		for _ in range(whole):
			step = q * self.excess(v)
			v_prev, v = v, v + step
			if step < 1e-6:
				break
		frac = k - whole
		if frac > 0:
			v_prev, v = v, v + frac * q * self.excess(v)
		return v, v_prev

	def score_bar(self, v: float) -> float:
		"""Lowest score whose utility reaches v (positive scores only)."""
		if v <= 0:
			return 0.0
		if self.ramp and v < self.ramp + self.lam_b:
			return v / (1 + self.lam_b / self.ramp)
		return max(0.0, v - self.lam_b)

	def pool_value(self, options: list[tuple[float, float]], future: int) -> float:
		"""Value of eventually marrying the best known partner who commits.

		options are (U, p_commit); we try them best first, one per round, and each
		may leave the market meanwhile at the observed singles hazard.
		"""
		if future <= 0 or not options:
			return 0.0
		options = sorted((o for o in options if o[0] > 0), reverse=True)[:future]
		rho = self.decay()
		value, miss = 0.0, 1.0
		for pos, (u, p) in enumerate(options):
			p_eff = p * math.exp(-rho * (pos + 1))
			value += miss * p_eff * u
			miss *= 1 - p_eff
			if miss < 1e-4:
				break
		return value
