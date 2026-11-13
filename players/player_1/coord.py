"""Coordination between copies of Player1, using only what the game shows.

Every copy works everything out on its own, from the ads and dates it sees:

1. Signature. A copy's ad is truthful about its own trait, and its seek half is
   a fixed function of that trait (``signature``). Copies recognise each other
   from the candidates' ads in round 1, and confirm it on the date: an honest
   copy's observed profile and stated ideal must reproduce its ad exactly.
2. Exploration. The copies, sorted by index, date by the circle method (a
   1-factorisation of the complete graph), so every pair meets once in n - 1
   rounds. Each copy lists only its scheduled partner, so the engine's mutual
   stage pairs them. Copies are honest, never ask again and never block each
   other, so nobody is held in a repeat date. Missed dates are retried.
3. Matching. Each copy now knows every copy's true profile and ideal, so it
   can compute everyone's happiness with everyone, and solves the SAME
   maximum-weight matching (deterministic input, deterministic algorithm).
4. Courting. A matched copy lists only its partner, asks again and proposes on
   the next date (date 2). Copies left unmatched, or whose partner does not
   respond, fall back to the normal search.

No state is shared between instances.
"""

from __future__ import annotations

from collections.abc import Callable

from models.player import Advertisement, Candidate, DateContext, DateOutcome, Decision
from models.profile import Profile, happiness

VALUE_SPAN = 11  # profile values 0..10


def signature(own_dim: int, own_value: int) -> tuple[int, int]:
	"""(seek_dim, seek_value) a copy advertises next to a truthful own trait.

	Chosen by measuring the ads of every other group (and the baselines): about
	0.7% of their seats produce this pattern by chance, and the date check
	rejects almost all of those.
	"""
	return own_dim, (own_value + 6) % VALUE_SPAN


def is_signed(ad: Advertisement) -> bool:
	return (ad.seek_dim, ad.seek_value) == signature(ad.own_dim, ad.own_value)


def circle_schedule(members: list[int]) -> list[dict[int, int]]:
	"""Round-robin rounds (circle method): every pair meets exactly once.

	``members`` must already be in the agreed order. With an odd count a bye
	(partner -1) is added. Returns, per round, a map member -> partner.
	"""
	people = list(members)
	if len(people) % 2:
		people.append(-1)
	n = len(people)
	rounds = []
	fixed, rest = people[0], people[1:]
	for r in range(n - 1):
		rot = rest[r:] + rest[:r]
		line = [fixed, *rot]
		pairs: dict[int, int] = {}
		for k in range(n // 2):
			a, b = line[k], line[n - 1 - k]
			pairs[a] = b
			pairs[b] = a
		pairs.pop(-1, None)
		rounds.append(pairs)
	return rounds


def copy_matching(
	members: list[int],
	info: dict[int, tuple[Profile, Profile]],
	bonus: float,
	monotonic: int,
	min_positive: int = 2,
	min_share: float = float('-inf'),
) -> dict[int, int]:
	"""The matching every copy computes: most positive seats, then most happiness.

	An edge (i, j) is allowed when at least ``min_positive`` of its two seats are
	positive and both sides get at least ``min_share * bonus``. Its weight is
	``L * positive seats + 2 * (h_ij + h_ji)``, rounded to an integer, with L large
	enough that one more positive seat beats any happiness difference.
	"""
	order = sorted(members)
	pos = {v: k for k, v in enumerate(order)}
	scale = int(8 * abs(bonus) + 8 * 400) + 1  # bounds |2 * (h_ij + h_ji)|
	big = 4 * scale
	edges = []
	for a in range(len(order)):
		for b in range(a + 1, len(order)):
			i, j = order[a], order[b]
			pi, ii = info[i]
			pj, ij = info[j]
			hi = happiness(bonus, pi, ii, pj, ij, monotonic)
			hj = happiness(bonus, pj, ij, pi, ii, monotonic)
			positive = (hi > 0) + (hj > 0)
			if positive < min_positive or min(hi, hj) < min_share * bonus:
				continue
			weight = big * positive + round(2 * (hi + hj))
			if weight > 0:
				edges.append((pos[i], pos[j], weight))
	mate = max_weight_matching(edges)
	out = {}
	for k, m in enumerate(mate):
		if m >= 0:
			out[order[k]] = order[m]
	return out


class Coordinator:
	"""One copy's view of the coordination. Owned by a single Player1 instance."""

	def __init__(
		self,
		index: int,
		profile: Profile,
		ideal: Profile,
		people: int,
		rounds: int,
		bonus: float,
		monotonic: int,
		expected_ad: Callable[[Profile, Profile], Advertisement],
		*,
		min_copies: int,
		max_copies: int,
		min_market_share: float,
		round_slack: int,
		retry_rounds: int,
		patience: int,
		min_positive: int,
		min_share: float,
	) -> None:
		self.index = index
		self.me = (tuple(profile), tuple(ideal))
		self.people = people
		self.rounds = rounds
		self.bonus = bonus
		self.monotonic = monotonic
		self.expected_ad = expected_ad
		self.min_copies = min_copies
		self.max_copies = max_copies
		self.min_market_share = min_market_share
		self.round_slack = round_slack
		self.retry_rounds = retry_rounds
		self.patience = patience
		self.min_positive = min_positive
		self.min_share = min_share

		self.state = 'new'  # new -> explore -> court -> done | off
		self.copies: list[int] = []
		self.schedule: list[dict[int, int]] = []
		self.info: dict[int, tuple[Profile, Profile]] = {index: self.me}
		self.fakes: set[int] = set()
		self.target: int | None = None
		self.matching: dict[int, int] = {}
		self.misses = 0

	@property
	def active(self) -> bool:
		return self.state in ('explore', 'court')

	def is_copy(self, partner: int) -> bool:
		return partner in self.copies and partner not in self.fakes

	# ----------------------------------------------------------- choose

	def choose(self, candidates: tuple[Candidate, ...], round_no: int) -> list[int] | None:
		"""Our interest list while coordinating, or None to let the normal search run."""
		if self.state == 'new':
			self._start(candidates, round_no)
		available = {c.index for c in candidates}
		if self.state == 'explore':
			step = round_no - 1
			if step < len(self.schedule):
				partner = self.schedule[step].get(self.index, -1)
				return [partner] if partner in available and partner not in self.info else []
			missing = [c for c in self._pending() if c in available]
			if missing and step < len(self.schedule) + self.retry_rounds:
				return missing
			self._solve()
		if self.state == 'court':
			if self.target in available:
				return [self.target]
			self.misses += 1
			if self.misses > self.patience:
				self.state = 'off'
				return None
			return []
		return None

	def _start(self, candidates: tuple[Candidate, ...], round_no: int) -> None:
		self.state = 'off'
		if round_no != 1:
			return
		copies = sorted({self.index} | {c.index for c in candidates if is_signed(c.ad)})
		n = len(copies)
		if n < self.min_copies or n > self.max_copies:
			return
		if n < self.min_market_share * self.people:
			return
		schedule = circle_schedule(copies)
		if self.rounds < len(schedule) + self.round_slack:
			return
		self.copies = copies
		self.schedule = schedule
		self.state = 'explore'

	def _pending(self) -> list[int]:
		return [c for c in self.copies if c not in self.info and c not in self.fakes]

	def _solve(self) -> None:
		if self._pending():
			# We never met some copy: we cannot compute the same matching as the rest.
			self.state = 'off'
			return
		members = [c for c in self.copies if c not in self.fakes]
		self.matching = copy_matching(
			members, self.info, self.bonus, self.monotonic, self.min_positive, self.min_share
		)
		self.target = self.matching.get(self.index)
		self.state = 'court' if self.target is not None else 'off'

	# ----------------------------------------------------------- dates

	def decide(
		self, date: DateContext, observed: Profile, stated_ideal: Profile, normal: Decision
	) -> Decision | None:
		"""Our decision while coordinating, or None to keep the normal one."""
		if not self.active:
			return None
		partner = date.partner
		if partner in self.copies and partner not in self.info and partner not in self.fakes:
			if self.expected_ad(tuple(observed), tuple(stated_ideal)) == date.partner_ad:
				self.info[partner] = (tuple(observed), tuple(stated_ideal))
			else:
				self.fakes.add(partner)
		if self.state == 'court' and partner == self.target:
			return Decision(again=True, propose=date.date_number >= 2)
		# Never block a copy (it would break the schedule); outsiders as usual.
		block = normal.incompatible and not self.is_copy(partner)
		return Decision(incompatible=block)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if self.state != 'court' or outcome.partner != self.target:
			return
		if outcome.married:
			self.state = 'done'
		elif outcome.date_number >= 2 and not outcome.partner_proposed:
			self.misses += 1
			if self.misses > self.patience:
				self.state = 'off'


# --------------------------------------------------------------------------
# Maximum-weight general matching (Edmonds' blossom algorithm, O(n^3)).
#
# A port of Joris van Rantwijk's public-domain mwmatching.py, trimmed to
# integer weights and no maximum-cardinality option.


def max_weight_matching(edges: list[tuple[int, int, int]]) -> list[int]:
	"""Maximum-weight matching of an undirected graph.

	``edges`` are (i, j, weight) with integer vertex ids from 0 and integer
	weights. Returns ``mate`` with mate[v] = partner of v, or -1.
	"""
	if not edges:
		return []
	nedge = len(edges)
	nvertex = 1 + max(max(i, j) for i, j, _ in edges)
	maxweight = max(0, max(w for _, _, w in edges))
	endpoint = [edges[p // 2][p % 2] for p in range(2 * nedge)]
	neighbend: list[list[int]] = [[] for _ in range(nvertex)]
	for k, (i, j, _) in enumerate(edges):
		neighbend[i].append(2 * k + 1)
		neighbend[j].append(2 * k)
	mate = nvertex * [-1]
	label = (2 * nvertex) * [0]
	labelend = (2 * nvertex) * [-1]
	inblossom = list(range(nvertex))
	blossomparent = (2 * nvertex) * [-1]
	blossomchilds: list = (2 * nvertex) * [None]
	blossombase = list(range(nvertex)) + nvertex * [-1]
	blossomendps: list = (2 * nvertex) * [None]
	bestedge = (2 * nvertex) * [-1]
	blossombestedges: list = (2 * nvertex) * [None]
	unusedblossoms = list(range(nvertex, 2 * nvertex))
	dualvar = nvertex * [maxweight] + nvertex * [0]
	allowedge = nedge * [False]
	queue: list[int] = []

	def slack(k: int) -> int:
		i, j, wt = edges[k]
		return dualvar[i] + dualvar[j] - 2 * wt

	def leaves(b: int):
		if b < nvertex:
			yield b
		else:
			for t in blossomchilds[b]:
				if t < nvertex:
					yield t
				else:
					yield from leaves(t)

	def assign_label(w: int, t: int, p: int) -> None:
		b = inblossom[w]
		label[w] = label[b] = t
		labelend[w] = labelend[b] = p
		bestedge[w] = bestedge[b] = -1
		if t == 1:
			queue.extend(leaves(b))
		elif t == 2:
			base = blossombase[b]
			assign_label(endpoint[mate[base]], 1, mate[base] ^ 1)

	def scan_blossom(v: int, w: int) -> int:
		path = []
		base = -1
		while v != -1 or w != -1:
			b = inblossom[v]
			if label[b] & 4:
				base = blossombase[b]
				break
			path.append(b)
			label[b] = 5
			if labelend[b] == -1:
				v = -1
			else:
				v = endpoint[labelend[b]]
				b = inblossom[v]
				v = endpoint[labelend[b]]
			if w != -1:
				v, w = w, v
		for b in path:
			label[b] = 1
		return base

	def add_blossom(base: int, k: int) -> None:
		v, w, _ = edges[k]
		bb = inblossom[base]
		bv = inblossom[v]
		bw = inblossom[w]
		b = unusedblossoms.pop()
		blossombase[b] = base
		blossomparent[b] = -1
		blossomparent[bb] = b
		path: list[int] = []
		endps: list[int] = []
		blossomchilds[b] = path
		blossomendps[b] = endps
		while bv != bb:
			blossomparent[bv] = b
			path.append(bv)
			endps.append(labelend[bv])
			v = endpoint[labelend[bv]]
			bv = inblossom[v]
		path.append(bb)
		path.reverse()
		endps.reverse()
		endps.append(2 * k)
		while bw != bb:
			blossomparent[bw] = b
			path.append(bw)
			endps.append(labelend[bw] ^ 1)
			w = endpoint[labelend[bw]]
			bw = inblossom[w]
		label[b] = 1
		labelend[b] = labelend[bb]
		dualvar[b] = 0
		for v in leaves(b):
			if label[inblossom[v]] == 2:
				queue.append(v)
			inblossom[v] = b
		bestedgeto = (2 * nvertex) * [-1]
		for bv in path:
			if blossombestedges[bv] is None:
				nblists = [[p // 2 for p in neighbend[v]] for v in leaves(bv)]
			else:
				nblists = [blossombestedges[bv]]
			for nblist in nblists:
				for k in nblist:
					i, j, _ = edges[k]
					if inblossom[j] == b:
						i, j = j, i
					bj = inblossom[j]
					if (
						bj != b
						and label[bj] == 1
						and (bestedgeto[bj] == -1 or slack(k) < slack(bestedgeto[bj]))
					):
						bestedgeto[bj] = k
			blossombestedges[bv] = None
			bestedge[bv] = -1
		blossombestedges[b] = [k for k in bestedgeto if k != -1]
		bestedge[b] = -1
		for k in blossombestedges[b]:
			if bestedge[b] == -1 or slack(k) < slack(bestedge[b]):
				bestedge[b] = k

	def expand_blossom(b: int, endstage: bool) -> None:
		for s in blossomchilds[b]:
			blossomparent[s] = -1
			if s < nvertex:
				inblossom[s] = s
			elif endstage and dualvar[s] == 0:
				expand_blossom(s, endstage)
			else:
				for v in leaves(s):
					inblossom[v] = s
		if (not endstage) and label[b] == 2:
			entrychild = inblossom[endpoint[labelend[b] ^ 1]]
			j = blossomchilds[b].index(entrychild)
			if j & 1:
				j -= len(blossomchilds[b])
				jstep = 1
				endptrick = 0
			else:
				jstep = -1
				endptrick = 1
			p = labelend[b]
			while j != 0:
				label[endpoint[p ^ 1]] = 0
				label[endpoint[blossomendps[b][j - endptrick] ^ endptrick ^ 1]] = 0
				assign_label(endpoint[p ^ 1], 2, p)
				allowedge[blossomendps[b][j - endptrick] // 2] = True
				j += jstep
				p = blossomendps[b][j - endptrick] ^ endptrick
				allowedge[p // 2] = True
				j += jstep
			bv = blossomchilds[b][j]
			label[endpoint[p ^ 1]] = label[bv] = 2
			labelend[endpoint[p ^ 1]] = labelend[bv] = p
			bestedge[bv] = -1
			j += jstep
			while blossomchilds[b][j] != entrychild:
				bv = blossomchilds[b][j]
				if label[bv] == 1:
					j += jstep
					continue
				v = -1
				for v in leaves(bv):
					if label[v] != 0:
						break
				if label[v] != 0:
					label[v] = 0
					label[endpoint[mate[blossombase[bv]]]] = 0
					assign_label(v, 2, labelend[v])
				j += jstep
		label[b] = labelend[b] = -1
		blossomchilds[b] = blossomendps[b] = None
		blossombase[b] = -1
		blossombestedges[b] = None
		bestedge[b] = -1
		unusedblossoms.append(b)

	def augment_blossom(b: int, v: int) -> None:
		t = v
		while blossomparent[t] != b:
			t = blossomparent[t]
		if t >= nvertex:
			augment_blossom(t, v)
		i = j = blossomchilds[b].index(t)
		if i & 1:
			j -= len(blossomchilds[b])
			jstep = 1
			endptrick = 0
		else:
			jstep = -1
			endptrick = 1
		while j != 0:
			j += jstep
			t = blossomchilds[b][j]
			p = blossomendps[b][j - endptrick] ^ endptrick
			if t >= nvertex:
				augment_blossom(t, endpoint[p])
			j += jstep
			t = blossomchilds[b][j]
			if t >= nvertex:
				augment_blossom(t, endpoint[p ^ 1])
			mate[endpoint[p]] = p ^ 1
			mate[endpoint[p ^ 1]] = p
		blossomchilds[b] = blossomchilds[b][i:] + blossomchilds[b][:i]
		blossomendps[b] = blossomendps[b][i:] + blossomendps[b][:i]
		blossombase[b] = blossombase[blossomchilds[b][0]]

	def augment_matching(k: int) -> None:
		v, w, _ = edges[k]
		for s, p in ((v, 2 * k + 1), (w, 2 * k)):
			while True:
				bs = inblossom[s]
				if bs >= nvertex:
					augment_blossom(bs, s)
				mate[s] = p
				if labelend[bs] == -1:
					break
				t = endpoint[labelend[bs]]
				bt = inblossom[t]
				s = endpoint[labelend[bt]]
				j = endpoint[labelend[bt] ^ 1]
				if bt >= nvertex:
					augment_blossom(bt, j)
				mate[j] = labelend[bt]
				p = labelend[bt] ^ 1

	for _ in range(nvertex):
		label[:] = (2 * nvertex) * [0]
		bestedge[:] = (2 * nvertex) * [-1]
		blossombestedges[nvertex:] = nvertex * [None]
		allowedge[:] = nedge * [False]
		queue[:] = []
		for v in range(nvertex):
			if mate[v] == -1 and label[inblossom[v]] == 0:
				assign_label(v, 1, -1)
		augmented = False
		while True:
			while queue and not augmented:
				v = queue.pop()
				for p in neighbend[v]:
					k = p // 2
					w = endpoint[p]
					if inblossom[v] == inblossom[w]:
						continue
					kslack = 0
					if not allowedge[k]:
						kslack = slack(k)
						if kslack <= 0:
							allowedge[k] = True
					if allowedge[k]:
						if label[inblossom[w]] == 0:
							assign_label(w, 2, p ^ 1)
						elif label[inblossom[w]] == 1:
							base = scan_blossom(v, w)
							if base >= 0:
								add_blossom(base, k)
							else:
								augment_matching(k)
								augmented = True
								break
						elif label[w] == 0:
							label[w] = 2
							labelend[w] = p ^ 1
					elif label[inblossom[w]] == 1:
						b = inblossom[v]
						if bestedge[b] == -1 or kslack < slack(bestedge[b]):
							bestedge[b] = k
					elif label[w] == 0 and (bestedge[w] == -1 or kslack < slack(bestedge[w])):
						bestedge[w] = k
			if augmented:
				break
			deltatype = 1
			delta = min(dualvar[:nvertex])
			deltaedge = deltablossom = -1
			for v in range(nvertex):
				if label[inblossom[v]] == 0 and bestedge[v] != -1:
					d = slack(bestedge[v])
					if d < delta:
						delta = d
						deltatype = 2
						deltaedge = bestedge[v]
			for b in range(2 * nvertex):
				if blossomparent[b] == -1 and label[b] == 1 and bestedge[b] != -1:
					d = slack(bestedge[b]) // 2
					if d < delta:
						delta = d
						deltatype = 3
						deltaedge = bestedge[b]
			for b in range(nvertex, 2 * nvertex):
				if (
					blossombase[b] >= 0
					and blossomparent[b] == -1
					and label[b] == 2
					and dualvar[b] < delta
				):
					delta = dualvar[b]
					deltatype = 4
					deltablossom = b
			for v in range(nvertex):
				if label[inblossom[v]] == 1:
					dualvar[v] -= delta
				elif label[inblossom[v]] == 2:
					dualvar[v] += delta
			for b in range(nvertex, 2 * nvertex):
				if blossombase[b] >= 0 and blossomparent[b] == -1:
					if label[b] == 1:
						dualvar[b] += delta
					elif label[b] == 2:
						dualvar[b] -= delta
			if deltatype == 1:
				break
			if deltatype == 2:
				allowedge[deltaedge] = True
				i, j, _ = edges[deltaedge]
				if label[inblossom[i]] == 0:
					i, j = j, i
				queue.append(i)
			elif deltatype == 3:
				allowedge[deltaedge] = True
				i, j, _ = edges[deltaedge]
				queue.append(i)
			elif deltatype == 4:
				expand_blossom(deltablossom, False)
		if not augmented:
			break
		for b in range(nvertex, 2 * nvertex):
			if blossomparent[b] == -1 and blossombase[b] >= 0 and label[b] == 1 and dualvar[b] == 0:
				expand_blossom(b, True)

	for v in range(nvertex):
		if mate[v] >= 0:
			mate[v] = endpoint[mate[v]]
	return mate
