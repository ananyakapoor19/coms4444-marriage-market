"""Experimental Player10 policies for the five-hour campaign.

This module does not replace the submitted player. Every optional decision
rule is disabled by default so the reference policy matches Player10 exactly.
Only public game parameters and the player's own observations are used.
"""

from __future__ import annotations

import math

from models.player import Decision
from players.player_10.player import Player10

EXTRA_DEFAULTS = {
	'singles_exponent': 0.0,
	'rank_known': False,
	'fix_exploration': False,
	'reserve_fresh': 0.0,
	'repeat_factor': 1.0,
	'reciprocal_factor': 1.0,
	'deception_penalty': 0.0,
	'population_rule': 'none',
}


class CampaignPlayer(Player10):
	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		for key, value in EXTRA_DEFAULTS.items():
			setattr(self, key, value)
		self.observations = {}
		self.inconsistency = {}

	def is_collect_data(self, ctx):
		if not self.fix_exploration:
			return super().is_collect_data(ctx)
		return (
			ctx.round < self.exploration_duration * (ctx.round + ctx.rounds_left)
			and len(self.estimates) < self.people - 1
		)

	def choose_dates(self, candidates, turn):
		if not (self.fix_exploration or self.rank_known or self.reserve_fresh):
			return super().choose_dates(candidates, turn)
		self.singles_left = turn.singles
		self.remaining = [c.index for c in candidates]
		if self.is_collect_data(turn):
			if self.fix_exploration:
				chosen = [c.index for c in candidates if c.index not in self.estimates]
			else:
				chosen = [c.index for c in candidates if c not in self.dated]
			return chosen[: self.list_limit]
		known = [c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0]
		if self.rank_known:
			known.sort(key=lambda i: (-self.estimates[i], i))
		fresh = sorted(
			(c for c in candidates if c.index not in self.estimates),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		fresh_count = max(1, int(len(fresh) * self.fresh_candidate_fraction))
		fresh_ids = [c.index for c in fresh[:fresh_count]]
		if not self.reserve_fresh:
			return (known + fresh_ids)[: self.list_limit]
		limit = self.list_limit if self.list_limit is not None else len(candidates)
		reserved = min(len(fresh_ids), max(1, int(limit * self.reserve_fresh)))
		known_count = max(0, limit - reserved)
		chosen = known[:known_count] + fresh_ids[:reserved]
		chosen += known[known_count:] + fresh_ids[reserved:]
		return chosen[:limit]

	def threshold(self, date):
		if self.decay_by_singles:
			remaining = self.singles_left / self.people
		else:
			remaining = date.rounds_left / self.rounds
		threshold = remaining**self.decay_exponent * self.bonus * self.threshold_scale
		if self.singles_exponent:
			threshold *= (self.singles_left / self.people) ** self.singles_exponent
		if self.population_rule == 'short_horizon':
			threshold *= min(1.0, self.rounds / max(2.0, math.sqrt(self.people)))
		elif self.population_rule == 'score_budget':
			threshold *= min(
				1.0, self.bonus / max(1.0, 12 * (self.dims - self.monotonic) + 6 * self.monotonic)
			)
		if date.partner in self.proposed_to_me:
			threshold *= self.reciprocal_factor
		return threshold

	def decide(self, date, observed, stated_ideal):
		if self.fix_exploration and date.partner not in self.dated:
			self.dated.append(date.partner)
		if not (
			self.singles_exponent
			or self.repeat_factor != 1
			or self.reciprocal_factor != 1
			or self.deception_penalty
			or self.population_rule != 'none'
		):
			return super().decide(date, observed, stated_ideal)
		estimate = self.calculate_estimate(observed, stated_ideal)
		if self.deception_penalty:
			ad = date.partner_ad
			evidence = (observed[ad.own_dim] - ad.own_value) ** 2
			evidence += 0.5 * (stated_ideal[ad.seek_dim] - ad.seek_value) ** 2
			if date.partner in self.observations:
				previous_profile, previous_ideal = self.observations[date.partner]
				evidence += sum(
					(a - b) ** 2 for a, b in zip(observed, previous_profile, strict=True)
				)
				evidence += 0.5 * sum(
					(a - b) ** 2 for a, b in zip(stated_ideal, previous_ideal, strict=True)
				)
			self.observations[date.partner] = (observed, stated_ideal)
			self.inconsistency[date.partner] = max(
				evidence, self.inconsistency.get(date.partner, 0)
			)
			estimate -= self.deception_penalty * self.inconsistency[date.partner]
		self.estimates[date.partner] = estimate
		if estimate <= 0:
			return Decision(incompatible=True)
		threshold = self.threshold(date)
		return Decision(
			again=estimate > threshold * self.repeat_factor,
			propose=date.date_number >= 2 and estimate > threshold,
		)


def make_player(parameters):
	class ConfiguredPlayer(CampaignPlayer):
		def __init__(self, snapshot, ctx):
			super().__init__(snapshot, ctx)
			for key, value in parameters.items():
				if key == 'monotonic_exaggeration':
					value = int(self.bonus / 75) if value == 'bonus_scaled' else value
				if not hasattr(self, key):
					raise ValueError(f'Unknown policy setting: {key}')
				setattr(self, key, value)

	ConfiguredPlayer.__name__ = 'Player10'
	return ConfiguredPlayer
