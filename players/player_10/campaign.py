"""Budgeted five-hour Player10 research campaign; does not edit Player10.

.venv/bin/python -m players.player_10.campaign
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

from players.player_10 import campaign_engine as engine
from players.player_10 import targeted_sweeps as design
from players.player_10.campaign_player import EXTRA_DEFAULTS

OUT = Path('results/player10_campaign')


def save(name, data):
	(OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def seconds_until(timestamp):
	return max(0, datetime.fromisoformat(timestamp).timestamp() - time.time())


def scenarios(seed, prefix, sizes):
	cfg = json.loads(design.DEFAULT_CONFIG.read_text())
	cases = design.make_scenarios(cfg, seed, prefix)
	rng = random.Random(seed + 1)
	population = (sizes * ((len(cases) + len(sizes) - 1) // len(sizes)))[: len(cases)]
	rng.shuffle(population)
	for s, n in zip(cases, population, strict=True):
		family = s['roster_family']
		if family == 'all_groups':
			seats = [str(i) for i in range(1, 11)] * max(2, n // 10)
		elif family == 'all_groups_random':
			seats = ([str(i) for i in range(1, 11)] + ['r']) * max(2, n // 11)
		elif family == 'half_half':
			opponent = next(c for c in s['seats'] if c != '10')
			seats = ['10'] * (n // 2) + [opponent] * (n // 2)
		elif family == 'random_heavy':
			ours = max(2, round(n * 0.1))
			seats = ['10'] * ours + ['r'] * (n - ours)
		else:
			seats = ['10'] * n
		s['seats'], s['people'] = seats, len(seats)
	return cases


def policy(parameters, label, kind):
	parameters = {**EXTRA_DEFAULTS, **parameters}
	key = json.dumps(parameters, sort_keys=True)
	return {
		'id': hashlib.sha256(key.encode()).hexdigest()[:12],
		'label': label,
		'kind': kind,
		'parameters': parameters,
	}


def unique(variants):
	return list({v['id']: v for v in variants}.values())


def rank(rows, variants):
	"""Compare only world/seed pairs completed by every candidate."""
	by = {v['id']: {} for v in variants}
	for r in rows:
		if r['variant'] in by and r['status'] == 'ok':
			by[r['variant']][(r['scenario'], r['seed'])] = r
	common = set.intersection(*(set(d) for d in by.values())) if by else set()
	if not common:
		return []
	result = []
	for v in variants:
		rs = [by[v['id']][key] for key in sorted(common)]
		result.append(
			{
				'variant': v['id'],
				'paired_games': len(rs),
				**{
					key: statistics.fmean(r[key] for r in rs)
					for key in (
						'positive_rate',
						'normalized_happiness',
						'mean_happiness',
						'bad_rate',
					)
				},
				'faults': sum(r['faults'] for r in rs),
			}
		)
	return sorted(
		result,
		key=lambda r: (r['positive_rate'], r['normalized_happiness'], r['variant']),
		reverse=True,
	)


def select(rows, variants, count):
	lookup = {v['id']: v for v in variants}
	return [lookup[r['variant']] for r in rank(rows, variants)[:count]]


def stage(name, cases, variants, seeds, budget, workers=10, cpu_limit=90):
	variants = unique(variants)
	save(f'{name}_design.json', {'scenarios': cases, 'variants': variants, 'seeds': seeds})
	rows = engine.run(name, cases, variants, seeds, OUT, workers, max(1, budget), cpu_limit)
	case_ids = {s['id'] for s in cases}
	variant_ids = {v['id'] for v in variants}
	rows = [
		r
		for r in rows
		if r['scenario'] in case_ids and r['variant'] in variant_ids and r['seed'] in seeds
	]
	ranking = rank(rows, variants)
	save(f'{name}_ranking.json', ranking)
	print(name, 'leaders:', json.dumps(ranking[:5]), flush=True)
	return rows, variants


def main():
	for key in (
		'OPENBLAS_NUM_THREADS',
		'OMP_NUM_THREADS',
		'MKL_NUM_THREADS',
		'VECLIB_MAXIMUM_THREADS',
	):
		os.environ[key] = '1'
	OUT.mkdir(exist_ok=True, parents=True)
	clock = json.loads((OUT / 'campaign_clock.json').read_text())

	def remaining():
		return seconds_until(clock['validation_start_utc'])

	refs = [policy(v['parameters'], v['label'], 'reference') for v in design.references()]
	prior = refs[1]['parameters']
	refs += [
		policy(
			{**prior, 'threshold_scale': 0.25, 'decay_exponent': exponent},
			f'Prior threshold finalist {exponent}',
			'reference',
		)
		for exponent in (2, 4)
	]
	source_files = [
		Path(__file__),
		Path(engine.__file__),
		Path(__file__).with_name('campaign_player.py'),
	]
	save(
		'source_hashes.json',
		{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
	)
	small = scenarios(50261112, 'search_small', [20])
	panel = design.balanced_subset(small, 32)
	rng = random.Random(50261113)
	choices = {
		'threshold_scale': [0, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1],
		'decay_exponent': [0.25, 0.5, 1, 2, 4, 8],
		'monotonic_exaggeration': [0, 1, 2, 3, 5, 'bonus_scaled'],
		'exploration_duration': [0, 0.1, 0.25, 0.5],
		'fresh_candidate_fraction': [0.25, 0.5, 0.75, 1],
	}
	sampled = unique(
		refs
		+ [
			policy(
				{**prior, **{k: rng.choice(v) for k, v in choices.items()}},
				f'Parameter sample {i}',
				'parameters',
			)
			for i in range(128)
		]
	)
	rows, candidates = stage(
		'parameter_screen', panel, sampled, [22001, 22002], min(2700, remaining()), workers=8
	)
	elites = unique(refs + select(rows, candidates, 16))
	rows, elites = stage(
		'parameter_confirm', small, elites, [22101, 22102, 22103], min(1800, remaining())
	)
	best = (select(rows, elites, 1) or [refs[1]])[0]
	ablations = [best, *refs]
	for key, values in {
		'rank_known': [True],
		'fix_exploration': [True],
		'reserve_fresh': [0.1, 0.25, 0.5],
		'repeat_factor': [0, 0.25, 0.5, 0.75],
		'reciprocal_factor': [0, 0.5, 0.75],
		'deception_penalty': [0.1, 0.25, 0.5, 1],
	}.items():
		for value in values:
			ablations.append(
				policy({**best['parameters'], key: value}, f'{key}={value}', 'ablation')
			)
	for duration in (0.1, 0.25, 0.5):
		ablations.append(
			policy(
				{**best['parameters'], 'fix_exploration': True, 'exploration_duration': duration},
				f'Fixed exploration {duration}',
				'ablation',
			)
		)
	ablation_cases = scenarios(50261114, 'ablation', [20, 20, 110])
	rows, ablations = stage(
		'ablations', ablation_cases, ablations, [23001, 23002, 23003], min(3000, remaining())
	)
	strong = select(rows, ablations, 5) or [best]
	combinations = [*strong, *refs, best]
	for i, a in enumerate(strong):
		for b in strong[i + 1 :]:
			changes = {k: v for k, v in b['parameters'].items() if v != best['parameters'].get(k)}
			combinations.append(
				policy(
					{**a['parameters'], **changes},
					f'Combined {a["label"]} + {b["label"]}',
					'combined',
				)
			)
	rows, combinations = stage(
		'combinations', ablation_cases, combinations, [23101, 23102, 23103], min(1800, remaining())
	)
	strong = select(rows, combinations, 3) or strong[:3]
	adaptive = [*strong, *refs]
	for base in strong[:2]:
		for beta in (0.5, 1, 2):
			adaptive.append(
				policy(
					{**base['parameters'], 'singles_exponent': beta},
					f'{base["label"]}; singles exponent {beta}',
					'hybrid',
				)
			)
		for rule in ('short_horizon', 'score_budget'):
			adaptive.append(
				policy(
					{**base['parameters'], 'population_rule': rule},
					f'{base["label"]}; {rule}',
					'population',
				)
			)
	rows, adaptive = stage(
		'adaptive', small, adaptive, [24001, 24002, 24003], min(1800, remaining())
	)
	finalists = unique(refs + (select(rows, adaptive, 4) or strong[:3]))
	large = scenarios(50261115, 'large', [110, 330, 1100])
	large = design.balanced_subset(large, 36)
	# Large-market evidence is a stress check, not a new source of hidden validation tuning.
	stage(
		'large_train', large, finalists, [25001], min(3600, remaining()), workers=4, cpu_limit=120
	)
	save(
		'frozen_finalists.json',
		{'variants': finalists, 'frozen_utc': datetime.now().astimezone().isoformat()},
	)
	print('FINALISTS FROZEN; beginning untouched validation', flush=True)
	validation = scenarios(60261112, 'validation', [20, 20, 110])
	validation_large = design.balanced_subset(
		scenarios(60261113, 'validation_large', [110, 330, 1100]), 30
	)
	# Fresh seeds and independently assigned worlds. No further tuning follows.
	budget = seconds_until(clock['hard_deadline_utc']) - 180
	stage(
		'validation',
		validation,
		finalists,
		list(range(31001, 31011)),
		min(3000, budget),
		workers=10,
		cpu_limit=90,
	)
	budget = seconds_until(clock['hard_deadline_utc']) - 180
	stage(
		'validation_large',
		validation_large,
		finalists,
		[32001, 32002, 32003],
		max(1, budget),
		workers=4,
		cpu_limit=120,
	)
	# Use remaining compute for a predeclared additional independent small/medium seed block.
	budget = seconds_until(clock['hard_deadline_utc']) - 180
	if budget > 60:
		stage(
			'validation_extra',
			validation,
			finalists,
			list(range(31101, 31121)),
			budget,
			workers=10,
			cpu_limit=90,
		)
	save(
		'finished.json',
		{
			'finished_utc': datetime.now().astimezone().isoformat(),
			'deadline_utc': clock['hard_deadline_utc'],
			'finalist_count': len(finalists),
		},
	)
	print('CAMPAIGN COMPLETE', flush=True)


if __name__ == '__main__':
	main()
