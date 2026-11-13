#!/usr/bin/env python3
"""Board tournament runner for the marriage market.

Expands the chalkboard axes into a pruned cell grid, runs every (cell, seed)
job in a process pool, checkpoints results as JSONL, and writes Project-1-style
CSV exports.

Examples:
    # Validate the full grid without running
    uv run tools/tournament_board.py --validate-only

    # Local smoke test
    uv run tools/tournament_board.py --limit 8 --workers 4 --out results/tournament_smoke

    # Shard across VMs (0/4 .. 3/4)
    uv run tools/tournament_board.py --shard 0/4 --workers 16 --out results/tournament_board

    # After every shard finishes, merge into one final set of 7 CSVs
    uv run tools/tournament_board.py --merge --out results/tournament_board
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from core.distributions import discover_distributions, warn_distribution_errors  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from models.profile import load_distribution_row, validate_distribution  # noqa: E402

GROUPS = tuple(str(i) for i in range(1, 11))
SEEDS = (1, 2, 3)
PAIR_SAMPLE = (
	('1', '2'),
	('3', '4'),
	('5', '6'),
	('7', '8'),
	('9', '10'),
	('1', '5'),
	('2', '6'),
	('3', '7'),
	('4', '9'),
)
_GROUP_FROM_NAME = re.compile(r'(?:Player)?(\d+)$', re.IGNORECASE)

DIST_FILES = {
	'u_shape': ROOT / 'players/player_9/distributions/u_shape.json',
	'right_skew': ROOT / 'players/player_2/distributions/right_skew.json',
	'p90_10': ROOT / 'players/player_10/distribution.json',
}


# ----------------------------------------------------------------- cells


@dataclass(frozen=True)
class Cell:
	cell_id: int
	slice: str
	roster: str
	label: str
	players: tuple[tuple[str, int], ...]
	rounds: int
	bonus: float
	dims: int
	monotonic: int
	distribution: str
	people: int

	def to_job(self, seed: int) -> dict[str, Any]:
		return {
			'cell_id': self.cell_id,
			'seed': seed,
			'slice': self.slice,
			'roster': self.roster,
			'label': self.label,
			'players': [list(p) for p in self.players],
			'rounds': self.rounds,
			'bonus': self.bonus,
			'dims': self.dims,
			'monotonic': self.monotonic,
			'distribution': self.distribution,
			'people': self.people,
			'timeout': 1.0,
			'list_multiple': 4.0,
		}


def _mono_modes(dims: int) -> list[tuple[str, int]]:
	modes = [('none', 0), ('all', dims)]
	if dims >= 2:
		modes.insert(1, ('half', dims // 2))
	return modes


def _roster_equal(n: int) -> tuple[tuple[str, int], ...]:
	if n % 10:
		raise ValueError(f'equal roster needs N divisible by 10, got {n}')
	k = n // 10
	return tuple((g, k) for g in GROUPS)


def _roster_equal_plus_random(group_each: int) -> tuple[tuple[str, int], ...]:
	"""k seats per group + 2 random (even substitute for board's +1 random)."""
	if group_each < 1:
		raise ValueError('group_each must be >= 1')
	return tuple((g, group_each) for g in GROUPS) + (('r', 2),)


def _roster_solo_random(group: str, n: int) -> tuple[tuple[str, int], ...]:
	if n % 2:
		raise ValueError(f'solo+random needs even N, got {n}')
	half = n // 2
	return ((group, half), ('r', half))


def _roster_pair(a: str, b: str, n: int) -> tuple[tuple[str, int], ...]:
	if n % 2:
		raise ValueError(f'half-half needs even N, got {n}')
	half = n // 2
	return ((a, half), (b, half))


def _roster_90_10(group: str, n: int) -> tuple[tuple[str, int], ...]:
	"""About 10% focal group, 90% random; both counts even, sum to n."""
	if n % 2:
		raise ValueError(f'90/10 needs even N, got {n}')
	g = max(2, (round(n * 0.1) // 2) * 2)
	g = min(g, n - 2)
	r = n - g
	if r % 2:
		g -= 1
		r += 1
	return ((group, g), ('r', r))


def _label(parts: dict[str, Any]) -> str:
	players = ' '.join(f'{c}x{n}' for c, n in parts['players'])
	return (
		f"{parts['slice']}|{parts['roster']}|B={parts['bonus']}|t={parts['rounds']}|"
		f"d={parts['dims']}|m={parts['monotonic']}|P={parts['distribution']}|"
		f"N={parts['people']}|{players}"
	)


def build_cells() -> list[Cell]:
	"""Pruned chalkboard grid. Order is deterministic; cell_id is the index."""
	raw: list[dict[str, Any]] = []

	def add(
		*,
		slice: str,
		roster: str,
		players: tuple[tuple[str, int], ...],
		rounds: int,
		bonus: float,
		dims: int,
		monotonic: int,
		distribution: str,
		**extra: Any,
	) -> None:
		people = sum(n for _, n in players)
		if people < 2 or people % 2:
			raise ValueError(f'illegal people={people} for {roster} {extra}')
		entry = {
			'slice': slice,
			'roster': roster,
			'players': players,
			'rounds': rounds,
			'bonus': bonus,
			'dims': dims,
			'monotonic': monotonic,
			'distribution': distribution,
			'people': people,
			**extra,
		}
		entry['label'] = _label(entry)
		raw.append(entry)

	core_B = (50, 100, 150)
	core_t = (50, 150)
	core_d = (2, 5, 10)
	core_P = ('uniform', 'normal', 'mixed')
	core_N = (20, 110)

	# A. Core × all roster kinds (full C(10,2) pairs)
	for bonus in core_B:
		for rounds in core_t:
			for dims in core_d:
				for _, monotonic in _mono_modes(dims):
					for dist in core_P:
						for n in core_N:
							base = dict(
								slice='A_core',
								rounds=rounds,
								bonus=bonus,
								dims=dims,
								monotonic=monotonic,
								distribution=dist,
							)
							add(roster='equal', players=_roster_equal(n), **base)
							# Board 10/11 & 20/22 → even 12 / 22 (k each + 2 random)
							if n == 20:
								add(
									roster='equal+random',
									players=_roster_equal_plus_random(2),
									**base,
								)
							elif n == 110:
								add(
									roster='equal+random',
									players=_roster_equal_plus_random(11),
									**base,
								)
							for g in GROUPS:
								add(
									roster='solo+random',
									players=_roster_solo_random(g, n),
									focal=g,
									**base,
								)
								add(
									roster='90random_10group',
									players=_roster_90_10(g, n),
									focal=g,
									**base,
								)
							for a, b in combinations(GROUPS, 2):
								add(
									roster='half-half',
									players=_roster_pair(a, b, n),
									pair=(a, b),
									**base,
								)

	# Micro equal / equal+random matching the board's 10/12 idea
	for bonus in core_B:
		for rounds in core_t:
			for dims in core_d:
				for _, monotonic in _mono_modes(dims):
					for dist in core_P:
						base = dict(
							slice='A_core_micro',
							rounds=rounds,
							bonus=bonus,
							dims=dims,
							monotonic=monotonic,
							distribution=dist,
						)
						add(roster='equal', players=_roster_equal(10), **base)
						add(roster='equal+random', players=_roster_equal_plus_random(1), **base)

	# B. Scale probes
	for n in (330, 1100):
		for rounds in (150, 500):
			for dims in (5, 10):
				base = dict(
					slice='B_scale',
					rounds=rounds,
					bonus=100,
					dims=dims,
					monotonic=dims // 2,
					distribution='mixed',
				)
				add(roster='equal', players=_roster_equal(n), **base)
				add(
					roster='equal+random',
					players=_roster_equal_plus_random(n // 10),
					**base,
				)
				for g in GROUPS:
					add(roster='solo+random', players=_roster_solo_random(g, n), focal=g, **base)
				for a, b in PAIR_SAMPLE:
					add(
						roster='half-half_sample',
						players=_roster_pair(a, b, n),
						pair=a + '+' + b,
						**base,
					)

	# C. Exotic distributions (m=half, compact rosters)
	exotic = ('u_shape', 'p90_10', 'p10_90', 'right_skew')
	for bonus in core_B:
		for rounds in core_t:
			for dims in core_d:
				monotonic = dims // 2
				for dist in exotic:
					for n in core_N:
						base = dict(
							slice='C_exotic',
							rounds=rounds,
							bonus=bonus,
							dims=dims,
							monotonic=monotonic,
							distribution=dist,
						)
						add(roster='equal', players=_roster_equal(n), **base)
						for g in GROUPS:
							add(
								roster='solo+random',
								players=_roster_solo_random(g, n),
								focal=g,
								**base,
							)
						for a, b in PAIR_SAMPLE:
							add(
								roster='half-half_sample',
								players=_roster_pair(a, b, n),
								pair=a + '+' + b,
								**base,
							)

	# D. Extremes: short/long t, tiny/huge d — equal + solos
	for bonus in (50, 100, 150, 500):
		for rounds, dims_list in ((5, (1, 2, 5)), (500, (5, 8, 10, 24))):
			for dims in dims_list:
				for _, monotonic in _mono_modes(dims):
					for n in (20, 110):
						base = dict(
							slice='D_extreme',
							rounds=rounds,
							bonus=bonus,
							dims=dims,
							monotonic=monotonic,
							distribution='mixed',
						)
						add(roster='equal', players=_roster_equal(n), **base)
						for g in GROUPS:
							add(
								roster='solo+random',
								players=_roster_solo_random(g, n),
								focal=g,
								**base,
							)

	# Stable dedupe
	seen: set[tuple] = set()
	cells: list[Cell] = []
	for entry in raw:
		key = (
			entry['bonus'],
			entry['rounds'],
			entry['dims'],
			entry['monotonic'],
			entry['distribution'],
			entry['roster'],
			entry['players'],
		)
		if key in seen:
			continue
		seen.add(key)
		cells.append(
			Cell(
				cell_id=len(cells),
				slice=entry['slice'],
				roster=entry['roster'],
				label=entry['label'],
				players=entry['players'],
				rounds=int(entry['rounds']),
				bonus=float(entry['bonus']),
				dims=int(entry['dims']),
				monotonic=int(entry['monotonic']),
				distribution=str(entry['distribution']),
				people=int(entry['people']),
			)
		)
	return cells


# ----------------------------------------------------------------- distributions


def _invert_row(row: tuple[float, ...]) -> tuple[float, ...]:
	return tuple(reversed(row))


def resolve_distribution(name: str, dims: int) -> tuple[Any, str]:
	"""Return an Engine distribution argument and a short name for logs."""
	if name in ('uniform', 'normal', 'mixed'):
		return name, name

	if name == 'p10_90':
		row = _invert_row(load_distribution_row(DIST_FILES['p90_10']))
		table = validate_distribution([row] * dims, dims)
		return table, 'p10_90'

	path = DIST_FILES.get(name)
	if path is None:
		raise ValueError(f'unknown distribution {name!r}')
	row = load_distribution_row(path)
	table = validate_distribution([row] * dims, dims)
	return table, name


def group_code(player_name: str) -> str:
	match = _GROUP_FROM_NAME.search(player_name.replace('_', ''))
	if match:
		return match.group(1)
	lower = player_name.lower()
	if 'random' in lower:
		return 'r'
	if 'greedy' in lower:
		return 'g'
	return player_name


# ----------------------------------------------------------------- workers


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
	"""Run one (cell, seed). Never raises into the pool."""
	started = time.time()
	try:
		available, _ = discover()
		roster: list = []
		for code, count in job['players']:
			if code not in available:
				raise KeyError(f'player {code!r} not loadable')
			roster.extend([available[code]] * int(count))

		if len(roster) != job['people'] or len(roster) % 2:
			raise ValueError(f'roster size {len(roster)} illegal for job people={job["people"]}')

		distribution, distribution_name = resolve_distribution(job['distribution'], job['dims'])
		random.seed(job['seed'])
		results = Engine(
			players=roster,
			rounds=job['rounds'],
			bonus=job['bonus'],
			dims=job['dims'],
			monotonic=job['monotonic'],
			seed=job['seed'],
			distribution=distribution,
			distribution_name=distribution_name,
			timeout=job.get('timeout', 1.0),
			keep_records=False,
			list_multiple=job.get('list_multiple', 4.0),
		).run()

		seats = []
		for person in results['players']:
			code = group_code(person['name'])
			seats.append(
				{
					'group': code,
					'index': person['index'],
					'married': person['married'],
					'happiness': person['happiness'],
					'positive': person['positive'],
					'partner': person['partner'],
					'married_on': person['married_on'],
				}
			)

		return {
			'ok': True,
			'error': None,
			'cell_id': job['cell_id'],
			'seed': job['seed'],
			'slice': job['slice'],
			'roster': job['roster'],
			'label': job['label'],
			'players': job['players'],
			'rounds': job['rounds'],
			'bonus': job['bonus'],
			'dims': job['dims'],
			'monotonic': job['monotonic'],
			'distribution': job['distribution'],
			'people': job['people'],
			'mean_happiness': results['mean_happiness'],
			'positive_rate': results['positive_rate'],
			'married_rate': results['married_rate'],
			'mean_married_happiness': results['mean_married_happiness'],
			'rounds_played': results['rounds_played'],
			'faults': len(results['faults']),
			'seats': seats,
			'elapsed_sec': round(time.time() - started, 4),
		}
	except Exception as exc:
		return {
			'ok': False,
			'error': f'{type(exc).__name__}: {exc}',
			'traceback': traceback.format_exc(limit=8),
			'cell_id': job.get('cell_id'),
			'seed': job.get('seed'),
			'label': job.get('label'),
			'elapsed_sec': round(time.time() - started, 4),
		}


# ----------------------------------------------------------------- io / aggregate


def _atomic_write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_suffix(path.suffix + '.tmp')
	tmp.write_text(json.dumps(payload, indent=2) + '\n')
	tmp.replace(path)


def load_done_keys(checkpoint: Path) -> set[tuple[int, int]]:
	done: set[tuple[int, int]] = set()
	if not checkpoint.exists():
		return done
	with checkpoint.open() as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			row = json.loads(line)
			if row.get('ok') and row.get('cell_id') is not None and row.get('seed') is not None:
				done.add((int(row['cell_id']), int(row['seed'])))
	return done


def append_checkpoint(checkpoint: Path, row: dict[str, Any]) -> None:
	checkpoint.parent.mkdir(parents=True, exist_ok=True)
	with checkpoint.open('a') as fh:
		fh.write(json.dumps(row, separators=(',', ':')) + '\n')
		fh.flush()
		os.fsync(fh.fileno())


def _mean(values: list[float]) -> float | None:
	return statistics.fmean(values) if values else None


def write_exports(out_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
	"""Project-1-style summary tables from successful runs."""
	out_dir.mkdir(parents=True, exist_ok=True)
	ok_rows = [r for r in rows if r.get('ok')]
	written: list[Path] = []

	raw_path = out_dir / '7_raw_runs.csv'
	with raw_path.open('w', newline='') as fh:
		writer = csv.writer(fh)
		writer.writerow(
			[
				'cell_id',
				'seed',
				'slice',
				'roster',
				'bonus',
				'rounds',
				'dims',
				'monotonic',
				'distribution',
				'people',
				'label',
				'seat',
				'group',
				'married',
				'happiness',
				'positive',
				'partner',
				'married_on',
				'market_mean_happiness',
				'market_positive_rate',
				'market_married_rate',
				'faults',
				'elapsed_sec',
			]
		)
		for row in ok_rows:
			for seat in row['seats']:
				writer.writerow(
					[
						row['cell_id'],
						row['seed'],
						row['slice'],
						row['roster'],
						row['bonus'],
						row['rounds'],
						row['dims'],
						row['monotonic'],
						row['distribution'],
						row['people'],
						row['label'],
						seat['index'],
						seat['group'],
						int(seat['married']),
						seat['happiness'],
						int(seat['positive']),
						seat['partner'],
						seat['married_on'],
						row['mean_happiness'],
						row['positive_rate'],
						row['married_rate'],
						row['faults'],
						row['elapsed_sec'],
					]
				)
	written.append(raw_path)

	# Overall standings by group (student groups only)
	by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in ok_rows:
		for seat in row['seats']:
			if seat['group'] in GROUPS:
				by_group[seat['group']].append(seat)

	standings_path = out_dir / '1_overall_standings.csv'
	with standings_path.open('w', newline='') as fh:
		writer = csv.writer(fh)
		writer.writerow(
			[
				'rank',
				'group',
				'seats',
				'positive_rate',
				'mean_happiness',
				'married_rate',
				'mean_married_happiness',
			]
		)
		ranked = []
		for group, seats in by_group.items():
			happy = [s['happiness'] for s in seats]
			married = [s for s in seats if s['married']]
			ranked.append(
				{
					'group': group,
					'seats': len(seats),
					'positive_rate': sum(1 for s in seats if s['positive']) / len(seats),
					'mean_happiness': _mean(happy),
					'married_rate': len(married) / len(seats),
					'mean_married_happiness': _mean([s['happiness'] for s in married]),
				}
			)
		ranked.sort(
			key=lambda r: (
				-(r['positive_rate'] or 0),
				-(r['mean_happiness'] or 0),
				r['group'],
			)
		)
		for i, row in enumerate(ranked, start=1):
			writer.writerow(
				[
					i,
					row['group'],
					row['seats'],
					f'{row["positive_rate"]:.6f}',
					f'{row["mean_happiness"]:.6f}' if row['mean_happiness'] is not None else '',
					f'{row["married_rate"]:.6f}',
					f'{row["mean_married_happiness"]:.6f}'
					if row['mean_married_happiness'] is not None
					else '',
				]
			)
	written.append(standings_path)

	def breakdown(path: Path, key: str) -> None:
		bucket: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
		for row in ok_rows:
			for seat in row['seats']:
				if seat['group'] not in GROUPS:
					continue
				bucket[(seat['group'], row[key])].append(seat)
		with path.open('w', newline='') as fh:
			writer = csv.writer(fh)
			writer.writerow(
				[
					'group',
					key,
					'seats',
					'positive_rate',
					'mean_happiness',
					'married_rate',
				]
			)
			for (group, value), seats in sorted(bucket.items(), key=lambda x: (x[0][0], str(x[0][1]))):
				writer.writerow(
					[
						group,
						value,
						len(seats),
						f'{sum(1 for s in seats if s["positive"]) / len(seats):.6f}',
						f'{_mean([s["happiness"] for s in seats]):.6f}',
						f'{sum(1 for s in seats if s["married"]) / len(seats):.6f}',
					]
				)
		written.append(path)

	breakdown(out_dir / '2_by_roster.csv', 'roster')
	breakdown(out_dir / '3_by_bonus.csv', 'bonus')
	breakdown(out_dir / '4_by_rounds.csv', 'rounds')
	breakdown(out_dir / '5_by_dims.csv', 'dims')
	breakdown(out_dir / '6_by_distribution.csv', 'distribution')

	errors = [r for r in rows if not r.get('ok')]
	err_path = out_dir / 'errors.jsonl'
	with err_path.open('w') as fh:
		for row in errors:
			fh.write(json.dumps(row) + '\n')
	written.append(err_path)

	meta = {
		'successful_runs': len(ok_rows),
		'failed_runs': len(errors),
		'groups_ranked': [r['group'] for r in ranked],
	}
	_atomic_write_json(out_dir / 'summary.json', meta)
	written.append(out_dir / 'summary.json')
	return written


# ----------------------------------------------------------------- validate / run


def validate_cells(cells: list[Cell], *, smoke: bool = True) -> None:
	available, errors = discover()
	for err in errors:
		print(f"warning: player '{err.code}' failed to load - {err.message}", file=sys.stderr)
	_, dist_errors = discover_distributions()
	warn_distribution_errors(dist_errors)

	required = set(GROUPS) | {'r'}
	missing = sorted(required - set(available))
	if missing:
		raise SystemExit(f'missing required players: {missing}')

	for name, path in DIST_FILES.items():
		if not path.is_file():
			raise SystemExit(f'distribution file for {name} missing: {path}')

	# Structural checks on every cell (cheap).
	for cell in cells:
		if cell.people < 2 or cell.people % 2:
			raise SystemExit(f'cell {cell.cell_id} has illegal N={cell.people}')
		if sum(n for _, n in cell.players) != cell.people:
			raise SystemExit(f'cell {cell.cell_id} players do not sum to people')
		if not (0 <= cell.monotonic <= cell.dims):
			raise SystemExit(f'cell {cell.cell_id} has illegal monotonic={cell.monotonic}')
		unknown = [code for code, _ in cell.players if code not in available]
		if unknown:
			raise SystemExit(f'cell {cell.cell_id} unknown players {unknown}')
		resolve_distribution(cell.distribution, cell.dims)

	if not smoke:
		return

	# One real Engine smoke per distinct (distribution, dims, monotonic).
	seen: set[tuple[str, int, int]] = set()
	for cell in cells:
		key = (cell.distribution, cell.dims, cell.monotonic)
		if key in seen:
			continue
		seen.add(key)
		distribution, distribution_name = resolve_distribution(cell.distribution, cell.dims)
		codes = [code for code, _ in cell.players]
		# Tiny even market using codes from this cell.
		probe = [available[codes[0]], available[codes[-1]]]
		Engine(
			players=probe,
			rounds=1,
			bonus=cell.bonus,
			dims=cell.dims,
			monotonic=cell.monotonic,
			distribution=distribution,
			distribution_name=distribution_name,
			seed=0,
			timeout=0,
			list_multiple=4.0,
			keep_records=False,
		).run()


def select_jobs(
	cells: list[Cell],
	*,
	seeds: tuple[int, ...],
	shard_index: int,
	shard_count: int,
	limit: int | None,
	done: set[tuple[int, int]],
) -> list[dict[str, Any]]:
	jobs: list[dict[str, Any]] = []
	for cell in cells:
		if cell.cell_id % shard_count != shard_index:
			continue
		for seed in seeds:
			if (cell.cell_id, seed) in done:
				continue
			jobs.append(cell.to_job(seed))
			if limit is not None and len(jobs) >= limit:
				return jobs
	return jobs


def run_pool(jobs: list[dict[str, Any]], workers: int, checkpoint: Path) -> list[dict[str, Any]]:
	results: list[dict[str, Any]] = []
	total = len(jobs)
	if total == 0:
		return results

	def handle(row: dict[str, Any]) -> None:
		append_checkpoint(checkpoint, row)
		results.append(row)
		n = len(results)
		status = 'ok' if row.get('ok') else 'ERR'
		print(
			f'\r  {n}/{total} {status} cell={row.get("cell_id")} seed={row.get("seed")}',
			end='',
			file=sys.stderr,
			flush=True,
		)
		if n == total:
			print(file=sys.stderr)

	if workers <= 1:
		for job in jobs:
			handle(_run_job(job))
		return results

	ctx = mp.get_context('spawn')
	with ctx.Pool(processes=workers) as pool:
		for row in pool.imap_unordered(_run_job, jobs, chunksize=1):
			handle(row)
	return results


def parse_shard(text: str) -> tuple[int, int]:
	if '/' not in text:
		raise argparse.ArgumentTypeError('shard must look like 0/4')
	left, right = text.split('/', 1)
	index, count = int(left), int(right)
	if count < 1 or index < 0 or index >= count:
		raise argparse.ArgumentTypeError(f'invalid shard {text}')
	return index, count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--out',
		type=Path,
		default=ROOT / 'results' / 'tournament_board',
		help='Output directory for checkpoints and CSV exports',
	)
	parser.add_argument('--workers', type=int, default=0, help='0 = cpu_count()')
	parser.add_argument('--shard', type=parse_shard, default=(0, 1), help='k/n shard of cells')
	parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
	parser.add_argument('--limit', type=int, default=None, help='Max jobs this invocation')
	parser.add_argument('--validate-only', action='store_true')
	parser.add_argument('--write-inventory', action='store_true', help='Rewrite inventory JSON/MD/JSONL')
	parser.add_argument(
		'--write-inventory-full',
		action='store_true',
		help='Also embed every cell inside pruned_inventory.json (large)',
	)
	parser.add_argument('--fresh', action='store_true', help='Ignore existing checkpoint')
	parser.add_argument(
		'--merge',
		action='store_true',
		help='Merge all checkpoint_shard*.jsonl under --out into one final set of 7 CSVs',
	)
	parser.add_argument(
		'--keep-shard-exports',
		action='store_true',
		help='Also write per-shard CSV folders (default: only checkpoints while running)',
	)
	return parser.parse_args(argv)


def load_checkpoint_rows(out_dir: Path) -> list[dict[str, Any]]:
	"""Load every shard checkpoint, deduping by (cell_id, seed); prefer ok rows."""
	paths = sorted(out_dir.glob('checkpoint_shard*.jsonl'))
	if not paths:
		raise SystemExit(f'no checkpoint_shard*.jsonl files under {out_dir}')

	best: dict[tuple[int, int], dict[str, Any]] = {}
	for path in paths:
		with path.open() as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				row = json.loads(line)
				if row.get('cell_id') is None or row.get('seed') is None:
					continue
				key = (int(row['cell_id']), int(row['seed']))
				prev = best.get(key)
				if prev is None:
					best[key] = row
					continue
				# Prefer successful rows over failures if both exist.
				if row.get('ok') and not prev.get('ok'):
					best[key] = row
	return [best[k] for k in sorted(best)]


def merge_exports(out_dir: Path, expected_runs: int | None = None) -> list[Path]:
	rows = load_checkpoint_rows(out_dir)
	ok = sum(1 for r in rows if r.get('ok'))
	bad = len(rows) - ok
	print(f'merge: {len(rows)} unique runs ({ok} ok, {bad} failed)', file=sys.stderr)
	if expected_runs is not None:
		print(f'merge: expected {expected_runs} total engine runs', file=sys.stderr)
		if ok < expected_runs:
			print(
				f'WARNING: only {ok}/{expected_runs} successful runs present',
				file=sys.stderr,
			)
	final_dir = out_dir / 'final'
	written = write_exports(final_dir, rows)
	# Convenience copies with the Project-1 style names at out_dir root.
	for path in list(written):
		if path.suffix == '.csv':
			target = out_dir / path.name
			target.write_bytes(path.read_bytes())
			written.append(target)
	return written


def write_inventory(cells: list[Cell], out_dir: Path, *, include_cells: bool = False) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	by_slice: dict[str, int] = defaultdict(int)
	by_roster: dict[str, int] = defaultdict(int)
	for cell in cells:
		by_slice[cell.slice] += 1
		by_roster[cell.roster] += 1
	payload: dict[str, Any] = {
		'seeds': list(SEEDS),
		'unique_cells': len(cells),
		'total_engine_runs': len(cells) * len(SEEDS),
		'by_slice': dict(by_slice),
		'by_roster': dict(by_roster),
		'notes': [
			'equal+random uses board 10/11 & 20/22 as 12/22 (k per group + 2 random).',
			'Full cell list is generated in-process by build_cells(); optional cells dump via --write-inventory-full.',
		],
	}
	if include_cells:
		payload['cells'] = [
			{
				'cell_id': c.cell_id,
				'slice': c.slice,
				'roster': c.roster,
				'label': c.label,
				'players': [list(p) for p in c.players],
				'rounds': c.rounds,
				'bonus': c.bonus,
				'dims': c.dims,
				'monotonic': c.monotonic,
				'distribution': c.distribution,
				'people': c.people,
			}
			for c in cells
		]
	_atomic_write_json(out_dir / 'pruned_inventory.json', payload)
	# Compact JSONL of cells for grep/shard tooling (streaming-friendly).
	cells_path = out_dir / 'cells.jsonl'
	with cells_path.open('w') as fh:
		for c in cells:
			fh.write(
				json.dumps(
					{
						'cell_id': c.cell_id,
						'slice': c.slice,
						'roster': c.roster,
						'players': [list(p) for p in c.players],
						'rounds': c.rounds,
						'bonus': c.bonus,
						'dims': c.dims,
						'monotonic': c.monotonic,
						'distribution': c.distribution,
						'people': c.people,
					},
					separators=(',', ':'),
				)
				+ '\n'
			)
	md = out_dir / 'CONFIG_LIST.md'
	md.write_text(
		'# Marriage Market Tournament — Config List (pruned)\n\n'
		f'- **Unique cells:** {len(cells)}\n'
		f'- **Engine runs (×{len(SEEDS)} seeds):** {len(cells) * len(SEEDS)}\n'
		f'- By slice: {dict(by_slice)}\n'
		f'- By roster: {dict(by_roster)}\n'
		'- equal+random uses board 10/11 & 20/22 as **12 / 22** '
		'(k per group + 2 random; engine requires even N).\n'
		'- Per-group seat counts: `cells.jsonl` → `players`.\n'
	)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	cells = build_cells()
	shard_index, shard_count = args.shard
	seeds = tuple(args.seeds)
	expected_runs = len(cells) * len(seeds)

	print(
		f'grid: {len(cells)} cells × {len(seeds)} seeds = {expected_runs} runs',
		file=sys.stderr,
	)

	if args.write_inventory or args.write_inventory_full:
		write_inventory(
			cells,
			ROOT / 'configs' / 'tournament_board',
			include_cells=bool(args.write_inventory_full),
		)
		print('wrote configs/tournament_board inventory', file=sys.stderr)

	if args.merge:
		written = merge_exports(args.out, expected_runs=expected_runs)
		print('wrote final 7 CSVs:')
		for path in written:
			if path.suffix == '.csv' and path.parent == args.out:
				print(f'  {path}')
		return 0

	print(f'shard: {shard_index}/{shard_count}', file=sys.stderr)

	if args.validate_only:
		print('validating distinct setups...', file=sys.stderr)
		validate_cells(cells)
		print('validation ok', file=sys.stderr)
		return 0

	out_dir = args.out
	out_dir.mkdir(parents=True, exist_ok=True)
	checkpoint = out_dir / f'checkpoint_shard{shard_index}of{shard_count}.jsonl'
	if args.fresh and checkpoint.exists():
		checkpoint.unlink()

	done = set() if args.fresh else load_done_keys(checkpoint)
	jobs = select_jobs(
		cells,
		seeds=seeds,
		shard_index=shard_index,
		shard_count=shard_count,
		limit=args.limit,
		done=done,
	)
	workers = args.workers or min(len(jobs) or 1, mp.cpu_count())
	print(
		f'queued: {len(jobs)} jobs (skipping {len(done)} done) on {workers} workers',
		file=sys.stderr,
	)

	# Fast preflight on this shard's distributions/players only
	if jobs:
		sample_cells = []
		seen_ids = set()
		by_id = {c.cell_id: c for c in cells}
		for job in jobs:
			cid = job['cell_id']
			if cid not in seen_ids:
				sample_cells.append(by_id[cid])
				seen_ids.add(cid)
			if len(sample_cells) >= 32:
				break
		validate_cells(sample_cells)

	started = time.time()
	new_rows = run_pool(jobs, workers, checkpoint)
	failed = sum(1 for r in new_rows if not r.get('ok'))
	print(
		f'finished shard work in {time.time() - started:.1f}s '
		f'({len(new_rows)} new, {failed} failed)',
		file=sys.stderr,
	)

	if args.keep_shard_exports:
		all_rows: list[dict[str, Any]] = []
		if checkpoint.exists():
			with checkpoint.open() as fh:
				for line in fh:
					line = line.strip()
					if line:
						all_rows.append(json.loads(line))
		written = write_exports(out_dir / f'shard{shard_index}of{shard_count}', all_rows)
		print('wrote shard exports:')
		for path in written:
			print(f'  {path}')
	else:
		print(f'checkpoint: {checkpoint}', file=sys.stderr)
		print('run with --merge after all shards finish for the final 7 CSVs', file=sys.stderr)

	if failed:
		print(f'WARNING: {failed} jobs failed; see checkpoint / --merge errors.jsonl', file=sys.stderr)
		return 2
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
