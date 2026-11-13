"""Full factorial Player10 search, followed by fresh-seed validation.

Run from the repository root:
    .venv/bin/python -m players.player_10.grid_search --jobs 10

Each variant replays identical profiles, ideals, seats, and random seeds.
Games are checkpointed to JSONL and resume automatically. Player10's shipped
defaults are never changed. Positive-happiness rate is the primary objective;
mean happiness divided by bonus breaks ties across equally weighted scenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from players.player_10.player import Player10  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'grid_search.json'
FAULT_INDEX = re.compile(r'^\S+?\[(\d+)\]')
METRICS = ('positive_rate', 'mean_happiness', 'normalized_happiness', 'married_rate', 'bad_rate')
AVAILABLE = None


def fingerprint() -> dict:
	paths = sorted(
		{
			*ROOT.glob('core/*.py'),
			*ROOT.glob('models/*.py'),
			*ROOT.glob('players/player_*/player.py'),
			*ROOT.glob('players/player_*/__init__.py'),
			ROOT / 'players/greedy_player.py',
			ROOT / 'players/random_player.py',
		}
	)
	return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def variants(cfg: dict) -> list[dict]:
	keys = list(cfg['parameters'])
	out = []
	for i, values in enumerate(product(*(cfg['parameters'][k] for k in keys))):
		out.append({'id': f'v{i:03d}', 'parameters': dict(zip(keys, values, strict=True))})
	return out


def effective(parameters: dict, bonus: float) -> dict:
	out = dict(parameters)
	if out['monotonic_exaggeration'] == 'bonus_scaled':
		out['monotonic_exaggeration'] = int(bonus / 75)
	return out


def baseline_id(all_variants: list[dict]) -> str:
	defaults = {
		'threshold_scale': 1.0,
		'decay_exponent': 1.0,
		'decay_by_singles': False,
		'monotonic_exaggeration': 'bonus_scaled',
		'exploration_duration': 0.5,
		'fresh_candidate_fraction': 0.5,
	}
	return next(v['id'] for v in all_variants if v['parameters'] == defaults)


def scenarios(cfg: dict) -> list[dict]:
	out = []
	for b, t, d, roster in product(cfg['bonuses'], cfg['rounds'], cfg['dims'], cfg['rosters']):
		seats = [code for code, n in roster['players'] for _ in range(n)]
		assert '10' in seats and len(seats) >= 2 and len(seats) % 2 == 0
		out.append(
			{
				'id': f's{len(out):03d}',
				'bonus': b,
				'rounds': t,
				'dims': d,
				'monotonic': min(2, d),
				'roster': roster['id'],
				'people': len(seats),
				'seats': seats,
			}
		)
	return out


def worker_init() -> None:
	global AVAILABLE
	AVAILABLE, errors = discover()
	if errors:
		raise RuntimeError(f'Player loading failed: {errors}')


def play(scenario: dict, seed: int, parameters: dict, cfg: dict) -> dict:
	# __init__ overrides are instance attributes in Player10, so apply sweep
	# settings after the normal initialization rather than as class attributes.
	settings = effective(parameters, scenario['bonus'])

	class Variant(Player10):
		def __init__(self, snapshot, ctx):
			super().__init__(snapshot, ctx)
			for name, value in settings.items():
				setattr(self, name, value)

	Variant.__name__ = 'Player10'
	seats = list(scenario['seats'])
	random.Random(f'player10-seats:{seed}').shuffle(seats)
	random.seed(seed)
	# Some opponent implementations use NumPy's global random stream.
	import numpy as np

	np.random.seed(seed)
	result = Engine(
		players=[Variant if code == '10' else AVAILABLE[code] for code in seats],
		rounds=scenario['rounds'],
		bonus=scenario['bonus'],
		dims=scenario['dims'],
		monotonic=scenario['monotonic'],
		seed=seed,
		distribution=cfg['distribution'],
		timeout=cfg['timeout'],
		list_multiple=cfg['list_multiple'],
		keep_records=False,
	).run()
	ours = [p for p in result['players'] if seats[p['index']] == '10']
	married = [p for p in ours if p['married']]
	ours_faults = []
	for fault in result['faults']:
		match = FAULT_INDEX.match(fault)
		if match and seats[int(match.group(1))] == '10':
			ours_faults.append(fault)
	mean = statistics.fmean(p['happiness'] for p in ours)
	return {
		'positive_rate': sum(p['happiness'] > 0 for p in ours) / len(ours),
		'mean_happiness': mean,
		'normalized_happiness': mean / scenario['bonus'],
		'married_rate': len(married) / len(ours),
		'bad_rate': sum(p['happiness'] <= 0 for p in married) / len(ours),
		'ours': len(ours),
		'positive': sum(p['happiness'] > 0 for p in ours),
		'married': len(married),
		'bad': sum(p['happiness'] <= 0 for p in married),
		'our_faults': len(ours_faults),
		'faults': len(result['faults']),
		'fault_examples': list(dict.fromkeys(result['faults']))[:3],
		'rounds_played': result['rounds_played'],
		'mean_married_round': statistics.fmean(p['married_on'] for p in married)
		if married
		else None,
	}


def run_batch(job: dict) -> list[dict]:
	rows = []
	for group in job['groups']:
		metrics = play(job['scenario'], job['seed'], group[0]['parameters'], job['cfg'])
		for v in group:
			rows.append(
				{
					'phase': job['phase'],
					'scenario': job['scenario']['id'],
					'seed': job['seed'],
					'variant': v['id'],
					'equivalent_to': group[0]['id'],
					**metrics,
				}
			)
	return rows


def load_rows(path: Path) -> dict[tuple, dict]:
	rows = {}
	if not path.exists():
		return rows
	# Drop a partial last line after interruption before appending more results.
	with path.open('r+b') as fh:
		good_end = 0
		while line := fh.readline():
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				if fh.read():
					raise ValueError('Corrupt checkpoint before last line') from None
				fh.truncate(good_end)
				break
			rows[(row['phase'], row['scenario'], row['seed'], row['variant'])] = row
			good_end = fh.tell()
	return rows


def run_phase(phase, cfg, markets, by_market, seeds, store, out, jobs):
	pending, logical, physical = [], 0, 0
	for s in markets:
		for seed in seeds:
			groups = defaultdict(list)
			for v in by_market[s['id']]:
				if (phase, s['id'], seed, v['id']) not in store:
					key = json.dumps(effective(v['parameters'], s['bonus']), sort_keys=True)
					groups[key].append(v)
			batches = list(groups.values())
			for start in range(0, len(batches), 8):
				batch = batches[start : start + 8]
				pending.append(
					{'phase': phase, 'scenario': s, 'seed': seed, 'groups': batch, 'cfg': cfg}
				)
				logical += sum(map(len, batch))
				physical += len(batch)
	# Interleave cheap and expensive scenarios rather than ending with all slow games.
	random.Random(12345).shuffle(pending)
	print(
		f'{phase}: {logical:,} pending grid rows, {physical:,} unique games; {jobs} workers',
		flush=True,
	)
	if not pending:
		return
	start, last, done, unique_done = time.monotonic(), 0.0, 0, 0
	ctx = mp.get_context('spawn')
	with (out / 'games.jsonl').open('a') as fh, ctx.Pool(jobs, initializer=worker_init) as pool:
		for rows in pool.imap_unordered(run_batch, pending, chunksize=1):
			for row in rows:
				fh.write(json.dumps(row, separators=(',', ':')) + '\n')
				store[(phase, row['scenario'], row['seed'], row['variant'])] = row
			fh.flush()
			done += len(rows)
			unique_done += len({r['equivalent_to'] for r in rows})
			now = time.monotonic()
			if now - last >= 15 or done == logical:
				elapsed = now - start
				eta = elapsed * (logical - done) / max(1, done)
				print(
					f'{phase}: {done:,}/{logical:,} rows; {unique_done:,} games; '
					f'{elapsed / 60:.1f} min elapsed; ~{eta / 60:.1f} min left',
					flush=True,
				)
				last = now


def summarize(rows: list[dict]) -> dict:
	return {
		'games': len(rows),
		**{m: statistics.fmean(r[m] for r in rows) for m in METRICS},
		'our_faults': sum(r['our_faults'] for r in rows),
		'faults': sum(r['faults'] for r in rows),
	}


def rank(stats: dict) -> tuple:
	return (round(stats['positive_rate'], 12), stats['normalized_happiness'])


def select(cfg, markets, all_variants, store) -> dict:
	by_variant, by_scenario = defaultdict(list), defaultdict(lambda: defaultdict(list))
	for (phase, s, _, v), row in store.items():
		if phase == 'grid':
			by_variant[v].append(row)
			by_scenario[s][v].append(row)
	expected = len(markets) * len(cfg['training_seeds'])
	assert all(len(by_variant[v['id']]) == expected for v in all_variants), 'Incomplete grid'
	stats = {v: summarize(rows) for v, rows in by_variant.items()}
	ordered = sorted(stats, key=lambda v: (*rank(stats[v]), v), reverse=True)
	champions = {}
	for s in markets:
		candidates = by_scenario[s['id']]
		assert all(len(candidates[v['id']]) == len(cfg['training_seeds']) for v in all_variants)
		champions[s['id']] = max(candidates, key=lambda v: (*rank(summarize(candidates[v])), v))
	return {
		'baseline': baseline_id(all_variants),
		'global_winner': ordered[0],
		'global_top3': ordered[:3],
		'scenario_winners': champions,
		'grid_ranking': [{'variant': v, **stats[v]} for v in ordered],
	}


def paired_interval(differences: list[float]) -> dict:
	mean = statistics.fmean(differences)
	half = (
		1.96 * statistics.stdev(differences) / math.sqrt(len(differences))
		if len(differences) > 1
		else 0
	)
	return {'mean': mean, 'low': mean - half, 'high': mean + half, 'n_seeds': len(differences)}


def validation_summary(cfg, markets, selected, store) -> dict:
	policies = {
		'baseline': None,
		**{v: v for v in selected['global_top3']},
		'scenario_selected': None,
	}
	out = {}
	for label, fixed in policies.items():
		rows, diffs = [], defaultdict(list)
		for s in markets:
			v = (
				selected['baseline']
				if label == 'baseline'
				else (
					selected['scenario_winners'][s['id']] if label == 'scenario_selected' else fixed
				)
			)
			for seed in cfg['validation_seeds']:
				r = store[('validation', s['id'], seed, v)]
				base = store[('validation', s['id'], seed, selected['baseline'])]
				rows.append(r)
				diffs[seed].append(r['positive_rate'] - base['positive_rate'])
		out[label] = {
			**summarize(rows),
			'positive_delta_vs_baseline': paired_interval(
				[statistics.fmean(ds) for ds in diffs.values()]
			),
		}
	return out


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--out', type=Path, default=ROOT / 'results/player10_grid')
	parser.add_argument('--jobs', type=int, default=min(10, os.cpu_count() or 1))
	parser.add_argument('--phase', choices=('all', 'grid', 'validation'), default='all')
	args = parser.parse_args()
	if args.jobs < 1:
		parser.error('--jobs must be positive')
	for variable in (
		'OPENBLAS_NUM_THREADS',
		'OMP_NUM_THREADS',
		'MKL_NUM_THREADS',
		'VECLIB_MAXIMUM_THREADS',
	):
		os.environ[variable] = '1'
	cfg = json.loads(args.config.read_text())
	all_variants, markets = variants(cfg), scenarios(cfg)
	manifest = {
		'schema': 1,
		'config': cfg,
		'source_hashes': fingerprint(),
		'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
		'variants': all_variants,
		'scenarios': markets,
	}
	args.out.mkdir(parents=True, exist_ok=True)
	manifest_path = args.out / 'experiment.json'
	if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
		raise SystemExit('Config or simulation source changed; use a new output directory.')
	manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
	store = load_rows(args.out / 'games.jsonl')
	print(
		f'{len(all_variants)} variants × {len(markets)} scenarios × '
		f'{len(cfg["training_seeds"])} seeds = '
		f'{len(all_variants) * len(markets) * len(cfg["training_seeds"]):,} grid rows',
		flush=True,
	)
	if args.phase in ('all', 'grid'):
		run_phase(
			'grid',
			cfg,
			markets,
			{s['id']: all_variants for s in markets},
			cfg['training_seeds'],
			store,
			args.out,
			args.jobs,
		)
	selection = select(cfg, markets, all_variants, store)
	(args.out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
	print('Grid winner:', selection['global_winner'], flush=True)
	if args.phase in ('all', 'validation'):
		lookup = {v['id']: v for v in all_variants}
		by_market = {
			s['id']: [
				lookup[v]
				for v in sorted(
					{
						selection['baseline'],
						*selection['global_top3'],
						selection['scenario_winners'][s['id']],
					}
				)
			]
			for s in markets
		}
		run_phase(
			'validation',
			cfg,
			markets,
			by_market,
			cfg['validation_seeds'],
			store,
			args.out,
			args.jobs,
		)
		validation = validation_summary(cfg, markets, selection, store)
		(args.out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
		print('Validation:', json.dumps(validation), flush=True)
	print(f'Results: {args.out}', flush=True)


if __name__ == '__main__':
	main()
