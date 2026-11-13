"""Robustness to consistent lies we cannot see on a date.

Three lies used by other groups leave no trace in a partner's ad or between dates:
- 'mask'  : regular traits clamped into [2, 8] (Player 4), so a shown 2 may hide a 0 or 1
            and a shown 8 a 9 or 10;
- 'nudge' : the trait our ad seeks moved up to 2 steps toward what we asked for (Player 8;
            monotonic traits only upward);
- 'blend' : a stated ideal pulled 65% of the way toward what we showed (Player 3, late game),
            which hides most of what we would cost them.

Each lie is a hypothesis about how the partner's report was produced. Its likelihood
against the honest one comes from the per-dimension value distribution we learn from first
dates (people's ideals are drawn from the same distribution as their traits). Learning is
itself corrupted by maskers - their 2s and 8s would make boundary values look common and
extremes rare - so the distribution of regular traits is fitted with a few EM steps that
treat every first-date report as possibly masked.

For every partner we return the posterior of each lie and two corrections to the score:
the expected extra cost (posterior-weighted, used for pricing) and the worst-case extra cost
under the lies we suspect (used to gate proposals).
"""

from models.profile import VALUES

LIES = ('mask', 'nudge', 'blend')
MASK_LOW, MASK_HIGH = 2, 8
NUDGE_STEP = 2
BLEND = 0.65
# Pseudo-count added to every value of every dimension's histogram.
SMOOTHING = 0.3
# Beta prior on the share of maskers, as pseudo-counts (masker, honest).
MASK_PRIOR_COUNTS = (1.0, 9.0)
EM_STEPS = 4
# After this many first dates, refit with one EM step from the previous fit.
WARM_AFTER = 40
# Weight of a stated ideal in the histogram, relative to a shown profile.
IDEAL_WEIGHT = 1.0


class LieModel:
	def __init__(
		self, player, lies=LIES, prior=0.1, suspect=0.5, price_floor=0.0, min_reports=0
	) -> None:
		self.p = player
		self.min_reports = min_reports  # below this many first dates, suspect nobody
		self.price_floor = price_floor  # lies less likely than this are not priced
		self.lies = set(lies)
		self.prior = prior
		self.suspect = suspect
		dims = player.dims
		self.regular = list(range(player.monotonic, dims))
		self.reports = 0  # first dates learned from
		# Histogram of first-date values: shown traits plus stated ideals, which are drawn
		# from the same distribution and not masked, so they anchor it while maskers are
		# being identified.
		self.counts = [[SMOOTHING] * len(VALUES) for _ in range(dims)]
		self.maskable: list[tuple[int, ...]] = []  # first-date reports a masker could give
		self._fit_size = -1
		self.dist: list[list[float]] = [_normalise(row) for row in self.counts]
		self.mask_share = prior

	# ------------------------------------------------------------- learning

	def learn(self, partner: int, observed, ad, stated_ideal) -> None:
		self.reports += 1
		for d, v in enumerate(observed):
			self.counts[d][v] += 1
		if tuple(stated_ideal) != tuple(self.p.profile):
			for d, v in enumerate(stated_ideal):
				self.counts[d][v] += IDEAL_WEIGHT
		if 'mask' in self.lies and self._maskable(observed, ad):
			self.maskable.append(tuple(observed))

	def _fit(self) -> None:
		"""Per-dimension value distributions, with masked reports unmasked by EM.

		Each refit starts from the last one, so once the sample is large a single step
		keeps it current (big markets refit after every first date).
		"""
		if self._fit_size == self.reports:
			return
		warm = self._fit_size >= WARM_AFTER
		self._fit_size = self.reports
		dist = self.dist if warm else [_normalise(row) for row in self.counts]
		share = self.mask_share
		if self.maskable and self.regular:
			a, b = MASK_PRIOR_COUNTS
			for _ in range(1 if warm else EM_STEPS):
				counts = [list(row) for row in self.counts]
				total = 0.0
				for shown in self.maskable:
					post = _posterior(share * self._mask_ratio(shown, dist), share)
					total += post
					for d in self.regular:
						v = shown[d]
						hidden = _hidden(v)
						if len(hidden) == 1:
							continue
						counts[d][v] -= post
						weight = sum(dist[d][t] for t in hidden)
						for t in hidden:
							counts[d][t] += post * dist[d][t] / weight
				dist = [_normalise(c) for c in counts]
				share = (total + a) / (self.reports + a + b)
		else:
			dist = [_normalise(row) for row in self.counts]
		self.dist = dist
		self.mask_share = share

	# -------------------------------------------------------------- scoring

	def assess(self, partner: int) -> tuple[float, float, dict[str, float]]:
		"""(expected extra cost, worst-case extra cost of suspected lies, posteriors)."""
		self._fit()
		p = self.p
		reports = p.reports[partner]
		ad = p.ads[partner]
		ratios: dict[str, float] = {}
		extra: dict[str, tuple[float, float]] = {}
		priors: dict[str, float] = {}

		if 'mask' in self.lies and self.regular:
			shown = reports[0][0]
			consistent = all(obs == shown for obs, _ in reports)
			if consistent and self._maskable(shown, ad):
				ratios['mask'] = self._mask_ratio(shown, self.dist)
				extra['mask'] = self._mask_extra(shown)
				priors['mask'] = self.mask_share

		if 'nudge' in self.lies:
			dim, want = p.my_ad.seek_dim, p.my_ad.seek_value
			values = {obs[dim] for obs, _ in reports}
			if len(values) == 1 and ad.own_dim != dim:
				s = values.pop()
				hidden = self._nudge_hidden(dim, want, s)
				if hidden:
					w = self.dist[dim]
					ratios['nudge'] = sum(w[t] for t in hidden) / w[s]
					extra['nudge'] = _extra(w, hidden, lambda t: p._gap(dim, p.ideal[dim], t), s)
					priors['nudge'] = self.prior

		if 'blend' in self.lies:
			ideals = [ideal for _, ideal in reports if tuple(ideal) != tuple(p.profile)]
			if ideals:
				# The ideal our score already uses: the one that costs us the most.
				ideal = max(ideals, key=lambda i: _cost(p, i))
				ratio, exp_extra, worst_extra = self._blend(ideal)
				if ratio > 0:
					ratios['blend'] = ratio
					extra['blend'] = (exp_extra / 2, worst_extra / 2)
					priors['blend'] = self.prior

		honest = 1.0 - sum(priors.values())
		norm = honest + sum(priors[h] * ratios[h] for h in ratios)
		post = {h: priors[h] * ratios[h] / norm for h in ratios}
		if self.reports < self.min_reports:
			return 0.0, 0.0, post
		expected = sum(post[h] * extra[h][0] for h in post if post[h] >= self.price_floor)
		worst = sum(extra[h][1] for h in post if post[h] >= self.suspect)
		return expected, worst, post

	# ---------------------------------------------------------------- masks

	def _maskable(self, shown, ad) -> bool:
		if any(not MASK_LOW <= shown[d] <= MASK_HIGH for d in self.regular):
			return False
		if ad.own_dim in self.regular and not MASK_LOW <= ad.own_value <= MASK_HIGH:
			return False
		return any(shown[d] in (MASK_LOW, MASK_HIGH) for d in self.regular)

	def _mask_ratio(self, shown, dist) -> float:
		ratio = 1.0
		for d in self.regular:
			hidden = _hidden(shown[d])
			if len(hidden) > 1:
				ratio *= sum(dist[d][t] for t in hidden) / dist[d][shown[d]]
		return ratio

	def _mask_extra(self, shown) -> tuple[float, float]:
		p = self.p
		exp = worst = 0.0
		for d in self.regular:
			hidden = _hidden(shown[d])
			if len(hidden) > 1:
				e, w = _extra(
					self.dist[d], hidden, lambda t, d=d: p._gap(d, p.ideal[d], t), shown[d]
				)
				exp += e
				worst += w
		return exp, worst

	# ---------------------------------------------------------------- nudge

	def nudge_worst(self, partner: int) -> float:
		"""Extra cost if this partner nudged the trait our ad seeks as far as it could.

		Unlike assess(), this ignores the posterior: a nudge cannot be told apart from
		honesty, so this is meant for a gate on everyone near zero.
		"""
		p = self.p
		dim, want = p.my_ad.seek_dim, p.my_ad.seek_value
		if p.ads[partner].own_dim == dim:
			return 0.0  # their ad already pins this trait down
		worst = 0.0
		for s in {obs[dim] for obs, _ in p.reports[partner]}:
			hidden = self._nudge_hidden(dim, want, s)
			if hidden:
				base = p._gap(dim, p.ideal[dim], s)
				worst = max(worst, max(p._gap(dim, p.ideal[dim], t) for t in hidden) - base)
		return worst

	def _nudge_hidden(self, dim: int, want: int, shown: int) -> list[int]:
		"""True values a seek-dim nudge could have turned into ``shown``."""
		if dim < self.p.monotonic:
			if shown > want:
				return []  # never nudged past the wish: shown is true
			if shown == want:
				return [t for t in range(want - NUDGE_STEP, want + 1) if t >= 0]
			return [shown - NUDGE_STEP] if shown - NUDGE_STEP >= 0 else []
		if shown == want:
			return [t for t in range(want - NUDGE_STEP, want + NUDGE_STEP + 1) if 0 <= t <= 10]
		t = shown - NUDGE_STEP if shown < want else shown + NUDGE_STEP
		return [t] if 0 <= t <= 10 else []

	# ---------------------------------------------------------------- blend

	def _blend(self, ideal) -> tuple[float, float, float]:
		"""(likelihood ratio, expected and worst extra cost to them) if ``ideal`` was blended."""
		p = self.p
		ratio = 1.0
		exp_cost = worst_cost = 0.0
		stated = _cost(p, ideal)
		for d, shown in enumerate(ideal):
			have = p.profile[d]
			hidden = [t for t in VALUES if _blended(t, have, d < p.monotonic) == shown]
			if not hidden:
				return 0.0, 0.0, 0.0
			w = self.dist[d]
			ratio *= sum(w[t] for t in hidden) / w[shown]
			costs = [p._gap(d, t, have) for t in hidden]
			weight = sum(w[t] for t in hidden)
			exp_cost += sum(w[t] * c for t, c in zip(hidden, costs, strict=True)) / weight
			worst_cost += max(costs)
		return ratio, exp_cost - stated, worst_cost - stated


def _blended(true: int, have: int, monotonic: bool) -> int:
	target = min(have, true) if monotonic else have
	return round((1 - BLEND) * true + BLEND * target)


def _hidden(v: int) -> list[int]:
	if v == MASK_LOW:
		return list(range(0, MASK_LOW + 1))
	if v == MASK_HIGH:
		return list(range(MASK_HIGH, 11))
	return [v]


def _extra(w, hidden, gap, shown) -> tuple[float, float]:
	"""Expected and worst extra gap if the truth is one of ``hidden`` rather than ``shown``."""
	weight = sum(w[t] for t in hidden)
	base = gap(shown)
	exp = sum(w[t] * gap(t) for t in hidden) / weight
	return exp - base, max(gap(t) for t in hidden) - base


def _cost(p, ideal) -> int:
	return sum(p._gap(d, want, p.profile[d]) for d, want in enumerate(ideal))


def _normalise(counts: list[float]) -> list[float]:
	total = sum(counts)
	return [c / total for c in counts]


def _posterior(weighted: float, prior: float) -> float:
	return weighted / (weighted + 1 - prior)
