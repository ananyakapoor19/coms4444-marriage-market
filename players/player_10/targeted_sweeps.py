"""Small evaluation-set coverage screen, threshold refinement, and validation.

    .venv/bin/python -m players.player_10.targeted_sweeps --jobs 10

This deliberately samples combinations of environment factors instead of
crossing the entire evaluation set. All Player10 agents share the swept
policy. The existing player and original full-grid experiment are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from models.profile import build_distribution, load_distribution_row
from players.player_10 import grid_search as grid

ROOT = grid.ROOT
DEFAULT_CONFIG = Path(__file__).parent / 'configs/targeted_sweeps.json'
FEATURES = ('distribution_name', 'dims', 'monotonic_mode', 'bonus', 'rounds', 'roster_family')


def references():
	defaults = {
		'threshold_scale': 1.0,
		'decay_exponent': 1.0,
		'decay_by_singles': False,
		'monotonic_exaggeration': 'bonus_scaled',
		'exploration_duration': 0.5,
		'fresh_candidate_fraction': 0.5,
	}
	winner = {
		**defaults,
		'threshold_scale': 0.5,
		'decay_exponent': 2.0,
		'monotonic_exaggeration': 2,
		'exploration_duration': 0.0,
	}
	return [
		{'id': 'defaults', 'label': 'Current defaults', 'parameters': defaults},
		{'id': 'previous_winner', 'label': 'Previous grid winner', 'parameters': winner},
		{
			'id': 'previous_safer',
			'label': 'Previous bonus-scaled candidate',
			'parameters': {**winner, 'monotonic_exaggeration': 'bonus_scaled'},
		},
	]


def distribution_rows():
	low = load_distribution_row(ROOT / 'players/player_10/distribution.json')
	return {
		'u_shape': load_distribution_row(ROOT / 'players/player_9/distribution.json'),
		'90_low_10_high': low,
		'10_low_90_high': tuple(reversed(low)),
		'right_skew': load_distribution_row(ROOT / 'players/player_2/distribution.json'),
	}


def make_scenarios(cfg, design_seed, prefix):
	rng = random.Random(design_seed)
	structures = list(product(cfg['dims'], cfg['monotonic_modes'], cfg['distributions']))
	worlds = list(product(cfg['bonuses'], cfg['rounds'], cfg['roster_families']))
	rng.shuffle(structures)
	rng.shuffle(worlds)
	# Each B × rounds × roster-family combination appears at least once.
	assigned = (worlds * ((len(structures) + len(worlds) - 1) // len(worlds)))[: len(structures)]
	rng.shuffle(assigned)
	duels = [str(i) for i in range(1, 10)] + ['r']
	rng.shuffle(duels)
	tables, out, duel_number = distribution_rows(), [], 0
	for i, ((dims, mode, name), (bonus, rounds, family)) in enumerate(
		zip(structures, assigned, strict=True)
	):
		monotonic = {'none': 0, 'half': dims // 2, 'all': dims}[mode]
		if family == 'all_groups':
			seats = [str(k) for k in range(1, 11)] * 2
		elif family == 'all_groups_random':
			seats = [str(k) for k in range(1, 11)] * 2 + ['r'] * 2
		elif family == 'half_half':
			seats = ['10'] * 10 + [duels[duel_number % len(duels)]] * 10
			duel_number += 1
		elif family == 'random_heavy':
			seats = ['10'] * 2 + ['r'] * 18
		elif family == 'self_play':
			seats = ['10'] * 20
		else:
			raise ValueError(f'Unknown roster family: {family}')
		table = (
			build_distribution(name, dims, monotonic)
			if name in ('uniform', 'normal', 'mixed')
			else [tables[name]] * dims
		)
		out.append(
			{
				'id': f'{prefix}{i:03d}',
				'bonus': bonus,
				'rounds': rounds,
				'dims': dims,
				'monotonic': monotonic,
				'monotonic_mode': mode,
				'distribution_name': name,
				'distribution': table,
				'roster_family': family,
				'roster': ' '.join(f'{code}x{n}' for code, n in sorted(Counter(seats).items())),
				'people': len(seats),
				'seats': seats,
			}
		)
	return out


def balanced_subset(scenarios, size):
	"""Choose a small, outcome-independent subset covering all feature values."""
	rng = random.Random(20261114)
	pool = list(scenarios)
	rng.shuffle(pool)
	counts, selected = Counter(), []
	while pool and len(selected) < size:
		case = max(
			pool,
			key=lambda s: (
				sum(counts[(f, s[f])] == 0 for f in FEATURES),
				-sum(counts[(f, s[f])] for f in FEATURES),
			),
		)
		selected.append(case)
		pool.remove(case)
		counts.update((f, case[f]) for f in FEATURES)
	return selected


def behavior_key(scenario, parameters):
	settings = grid.effective(parameters, scenario['bonus'])
	if settings['threshold_scale'] == 0:
		settings['decay_exponent'] = 1.0  # Any exponent gives threshold zero.
	if scenario['monotonic'] == 0:
		settings['monotonic_exaggeration'] = 0  # No traits can be exaggerated.
	return json.dumps(settings, sort_keys=True)


def simulation_id(scenario, seed, parameters):
	key = [scenario['id'], seed, behavior_key(scenario, parameters)]
	return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]


def worker(job):
	out = []
	for group in job['groups']:
		variant = group[0]
		metrics = grid.play(
			job['scenario'],
			job['seed'],
			variant['parameters'],
			{**job['cfg'], 'distribution': job['scenario']['distribution']},
		)
		for v in group:
			out.append(
				{
					'phase': job['phase'],
					'scenario': job['scenario']['id'],
					'seed': job['seed'],
					'variant': v['id'],
					'simulation_id': simulation_id(job['scenario'], job['seed'], v['parameters']),
					**metrics,
				}
			)
	return out


def run_phase(phase, cfg, scenarios, variants, seeds, store, out, jobs):
	cache = {r['simulation_id']: r for r in store.values()}
	pending, reused = [], []
	for scenario in scenarios:
		for seed in seeds:
			groups = defaultdict(list)
			for v in variants:
				if (phase, scenario['id'], seed, v['id']) in store:
					continue
				sim = simulation_id(scenario, seed, v['parameters'])
				if sim in cache:
					reused.append({**cache[sim], 'phase': phase, 'variant': v['id']})
				else:
					groups[sim].append(v)
			for group in groups.values():
				pending.append(
					{
						'phase': phase,
						'cfg': cfg,
						'scenario': scenario,
						'seed': seed,
						'groups': [group],
					}
				)
	random.Random(123).shuffle(pending)
	print(
		f'{phase}: {len(pending):,} new games; {len(reused):,} cached rows; '
		f'{len(scenarios)} scenarios × {len(variants)} variants × {len(seeds)} seeds',
		flush=True,
	)
	start, last, completed = time.monotonic(), 0, 0
	with (out / 'games.jsonl').open('a') as fh:

		def save(rows):
			for row in rows:
				fh.write(json.dumps(row, separators=(',', ':')) + '\n')
				store[(phase, row['scenario'], row['seed'], row['variant'])] = row
			fh.flush()

		save(reused)
		if pending:
			with mp.get_context('spawn').Pool(jobs, initializer=grid.worker_init) as pool:
				for rows in pool.imap_unordered(worker, pending, chunksize=1):
					save(rows)
					completed += 1
					now = time.monotonic()
					if now - last >= 15 or completed == len(pending):
						elapsed = now - start
						eta = elapsed / completed * (len(pending) - completed)
						print(
							f'{phase}: {completed:,}/{len(pending):,} games; '
							f'{elapsed / 60:.1f} min elapsed; ~{eta / 60:.1f} min left',
							flush=True,
						)
						last = now
	for s, seed, v in product(scenarios, seeds, variants):
		assert (phase, s['id'], seed, v['id']) in store


def ranking(phase, variants, store):
	out = []
	for v in variants:
		rows = [r for r in store.values() if r['phase'] == phase and r['variant'] == v['id']]
		out.append({'variant': v['id'], **grid.summarize(rows)})
	return sorted(out, key=lambda r: (*grid.rank(r), r['variant']), reverse=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--out', type=Path, default=ROOT / 'results/player10_targeted')
	parser.add_argument('--jobs', type=int, default=10)
	args = parser.parse_args()
	if args.jobs < 1:
		parser.error('--jobs must be positive')
	for name in (
		'OPENBLAS_NUM_THREADS',
		'OMP_NUM_THREADS',
		'MKL_NUM_THREADS',
		'VECLIB_MAXIMUM_THREADS',
	):
		os.environ[name] = '1'
	cfg = json.loads(args.config.read_text())
	coverage = make_scenarios(cfg, cfg['coverage_design_seed'], 'c')
	validation = make_scenarios(cfg, cfg['validation_design_seed'], 'v')
	refinement = balanced_subset(coverage, cfg['refinement_scenarios'])
	refs = references()
	manifest = json.loads(
		json.dumps(
			{
				'schema': 1,
				'config': cfg,
				'source_hashes': grid.fingerprint(),
				'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
				'game_runner_sha256': hashlib.sha256(Path(grid.__file__).read_bytes()).hexdigest(),
				'coverage_scenarios': coverage,
				'refinement_scenarios': [s['id'] for s in refinement],
				'validation_scenarios': validation,
				'reference_variants': refs,
			}
		)
	)
	args.out.mkdir(parents=True, exist_ok=True)
	path = args.out / 'experiment.json'
	if path.exists() and json.loads(path.read_text()) != manifest:
		raise SystemExit('Experiment changed; choose a new --out directory.')
	path.write_text(json.dumps(manifest, indent=2) + '\n')
	store = grid.load_rows(args.out / 'games.jsonl')
	run_phase('coverage', cfg, coverage, refs, cfg['training_seeds'], store, args.out, args.jobs)
	coverage_rank = ranking('coverage', refs, store)
	base = next(v for v in refs if v['id'] == coverage_rank[0]['variant'])
	thresholds = [
		{
			'id': f't{i:02d}',
			'label': f'Scale {scale:g}, exponent {exponent:g}',
			'parameters': {
				**base['parameters'],
				'threshold_scale': scale,
				'decay_exponent': exponent,
			},
		}
		for i, (scale, exponent) in enumerate(
			product(cfg['threshold_scales'], cfg['decay_exponents'])
		)
	]
	all_variants = refs + thresholds
	print('Coverage ranking:', json.dumps(coverage_rank), flush=True)
	print('Refinement base:', base['id'], flush=True)
	run_phase(
		'refinement',
		cfg,
		refinement,
		all_variants,
		cfg['training_seeds'],
		store,
		args.out,
		args.jobs,
	)
	refinement_rank = ranking('refinement', all_variants, store)
	# Freeze two behaviorally distinct threshold finalists before validation.
	lookup, finalists, seen = {v['id']: v for v in all_variants}, [], set()
	for r in refinement_rank:
		if not r['variant'].startswith('t'):
			continue
		v = lookup[r['variant']]
		key = json.dumps(v['parameters'], sort_keys=True)
		if v['parameters']['threshold_scale'] == 0:
			key = json.dumps({**v['parameters'], 'decay_exponent': 1.0}, sort_keys=True)
		if key not in seen:
			finalists.append(v)
			seen.add(key)
		if len(finalists) == 2:
			break
	selection = {
		'coverage_ranking': coverage_rank,
		'refinement_base': base['id'],
		'variants': all_variants,
		'refinement_ranking': refinement_rank,
		'finalists': [v['id'] for v in finalists],
	}
	(args.out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
	print('Frozen finalists:', [v['id'] for v in finalists], flush=True)
	run_phase(
		'validation',
		cfg,
		validation,
		refs + finalists,
		cfg['validation_seeds'],
		store,
		args.out,
		args.jobs,
	)
	summary = ranking('validation', refs + finalists, store)
	(args.out / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
	print('Validation:', json.dumps(summary), flush=True)
	print('Done:', args.out, flush=True)


if __name__ == '__main__':
	main()
