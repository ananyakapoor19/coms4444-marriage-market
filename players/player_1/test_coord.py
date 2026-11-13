import functools
import random

import pytest

import players.player_1.player as p1
from core.engine import Engine
from models.player import Advertisement
from players.player_1.coord import (
	circle_schedule,
	copy_matching,
	is_signed,
	max_weight_matching,
	signature,
)


def _brute(n: int, edges: list[tuple[int, int, int]]) -> int:
	weights = {(min(i, j), max(i, j)): w for i, j, w in edges}

	@functools.cache
	def best(mask: int) -> int:
		if not mask:
			return 0
		i = (mask & -mask).bit_length() - 1
		rest = mask & ~(1 << i)
		value = best(rest)
		for j in range(n):
			if rest >> j & 1 and (i, j) in weights:
				value = max(value, weights[(i, j)] + best(rest & ~(1 << j)))
		return value

	return best((1 << n) - 1)


def _weight(edges, mate) -> int:
	weights = {(min(i, j), max(i, j)): w for i, j, w in edges}
	return sum(weights[(v, m)] for v, m in enumerate(mate) if m > v)


def test_matching_hand_cases():
	# A path a-b-c-d: the two outer edges beat the heavier middle one.
	edges = [(0, 1, 5), (1, 2, 8), (2, 3, 5)]
	assert max_weight_matching(edges) == [1, 0, 3, 2]
	# A heavy middle edge wins when the outer ones are light.
	assert max_weight_matching([(0, 1, 2), (1, 2, 8), (2, 3, 2)]) == [-1, 2, 1, -1]
	# An odd cycle (a blossom) with a pendant vertex.
	edges = [(0, 1, 6), (1, 2, 6), (2, 0, 6), (2, 3, 5)]
	mate = max_weight_matching(edges)
	assert _weight(edges, mate) == 11
	assert mate[3] == 2
	assert max_weight_matching([]) == []


def test_matching_matches_brute_force():
	rng = random.Random(7)
	for _ in range(400):
		n = rng.randint(2, 9)
		edges = [
			(i, j, rng.randint(1, 50))
			for i in range(n)
			for j in range(i + 1, n)
			if rng.random() < 0.6
		]
		if not edges:
			continue
		mate = max_weight_matching(edges)
		for v, m in enumerate(mate):
			assert m == -1 or mate[m] == v
		assert _weight(edges, mate) == _brute(n, edges)


def test_copy_matching_prefers_more_positive_pairs():
	# Monotonic dims are 0..1; everyone wants (5,5,5) and person 0 is perfect.
	ideal = (5, 5, 5)
	info = {
		0: ((5, 5, 5), ideal),
		1: ((5, 5, 5), ideal),
		2: ((5, 5, 2), ideal),
		3: ((5, 5, 2), ideal),
	}
	# A gap of 3 on dim 2 costs 9 to whoever wants 5 there, and half of 9 to the
	# other side. B = 10: 0-2 gives 1 and 5.5, both positive, so two such pairs
	# (4 positive seats) beat the perfect 0-1 pair (2 positive seats).
	m = copy_matching([0, 1, 2, 3], info, 10.0, 2)
	assert len(m) == 4 and {m[0], m[1]} == {2, 3}
	# B = 8: 0-2 leaves 0 at -1, and 2-3 is negative, so only 0-1 remains.
	assert copy_matching([0, 1, 2, 3], info, 8.0, 2) == {0: 1, 1: 0}
	# 0-2 then has one positive seat (2 gets 3.5): allowed only with min_positive=1,
	# and even then 0-1 + nothing (2 seats, 16 happiness) beats 0-2 + 1-3 (2 seats, 5).
	assert copy_matching([0, 2], info, 8.0, 2, min_positive=2) == {}
	assert copy_matching([0, 2], info, 8.0, 2, min_positive=1) == {0: 2, 2: 0}
	assert copy_matching([0, 1, 2, 3], info, 8.0, 2, min_positive=1) == {0: 1, 1: 0}
	assert copy_matching([2, 3], info, 10.0, 2) == {}


def test_schedule_meets_everyone_once():
	for n in (2, 5, 10, 11, 20):
		members = list(range(3, 3 + n))
		seen = set()
		for rnd in circle_schedule(members):
			for a, b in rnd.items():
				if b == -1:  # bye
					assert n % 2
					continue
				assert rnd[b] == a and a != b
				seen.add((min(a, b), max(a, b)))
		assert len(seen) == n * (n - 1) // 2


def test_signature():
	for dim in range(5):
		for value in range(11):
			seek_dim, seek_value = signature(dim, value)
			assert is_signed(Advertisement(dim, value, seek_dim, seek_value))
			assert 0 <= seek_value <= 10


@pytest.fixture
def coordinate(monkeypatch):
	monkeypatch.setattr(p1, 'COORDINATE', True)


@pytest.mark.parametrize('bonus', [100, 200])
def test_solo_coordination(coordinate, bonus):
	positive = seats = 0
	for seed in (1, 2, 3):
		engine = Engine(
			players=[p1.Player1] * 20, rounds=100, bonus=bonus, seed=seed, keep_records=False
		)
		res = engine.run()
		assert res['faults'] == []
		positive += sum(p['positive'] for p in res['players'])
		seats += len(res['players'])
	assert positive / seats >= 0.98


def test_pairs_coordination_no_faults(coordinate):
	from players.greedy_player import GreedyPlayer

	engine = Engine(players=[p1.Player1] * 10 + [GreedyPlayer] * 10, rounds=100, bonus=100, seed=5)
	res = engine.run()
	assert res['faults'] == []


class _Impostor(p1.Player1):
	"""Posts our signature but lies about its own trait and ignores the schedule."""

	def project(self, date):
		own = self.my_ad.own_dim
		shown = list(self.profile)
		shown[own] = (shown[own] + 3) % 11
		return tuple(shown)

	def choose_dates(self, candidates, turn):
		return []


def test_impostors_are_rejected(coordinate):
	roster = [p1.Player1] * 9 + [_Impostor] + [p1.Player1] * 9 + [_Impostor]
	positive = 0
	for seed in (1, 2):
		engine = Engine(players=roster, rounds=100, bonus=200, seed=seed, keep_records=False)
		res = engine.run()
		assert res['faults'] == []
		positive += sum(p['positive'] for p in res['players'] if roster[p['index']] is p1.Player1)
	assert positive / 36 >= 0.9
