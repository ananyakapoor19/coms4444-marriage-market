"""Isolated, resumable campaign games with CPU and phase time limits."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import signal
import statistics
import time
from pathlib import Path

from core.engine import Engine
from core.registry import discover
from players.player_10.campaign_player import make_player

AVAILABLE = None


class GameBudget(BaseException):
	pass


def budget_expired(*args):
	raise GameBudget()


def initialize():
	global AVAILABLE
	AVAILABLE, errors = discover()
	if errors:
		raise RuntimeError(str(errors))


def play(job):
	import numpy as np

	s, v, seed = job['scenario'], job['variant'], job['seed']
	started, cpu = time.monotonic(), time.process_time()
	base = {'scenario': s['id'], 'variant': v['id'], 'seed': seed}
	previous = signal.signal(signal.SIGPROF, budget_expired)
	signal.setitimer(signal.ITIMER_PROF, job.get('cpu_limit', 180))
	try:
		seats = list(s['seats'])
		random.Random(f'campaign-seats:{seed}').shuffle(seats)
		random.seed(seed)
		np.random.seed(seed)
		cls = make_player(v['parameters'])
		result = Engine(
			players=[cls if c == '10' else AVAILABLE[c] for c in seats],
			rounds=s['rounds'],
			bonus=s['bonus'],
			dims=s['dims'],
			monotonic=s['monotonic'],
			seed=seed,
			distribution=s['distribution'],
			keep_records=False,
			timeout=1.0,
		).run()
		ours = [p for p in result['players'] if seats[p['index']] == '10']
		married = [p for p in ours if p['married']]
		mean = statistics.fmean(p['happiness'] for p in ours)
		return {
			**base,
			'status': 'ok',
			'positive_rate': sum(p['happiness'] > 0 for p in ours) / len(ours),
			'mean_happiness': mean,
			'normalized_happiness': mean / s['bonus'],
			'bad_rate': sum(p['happiness'] <= 0 for p in married) / len(ours),
			'married_rate': len(married) / len(ours),
			'ours': len(ours),
			'faults': len(result['faults']),
			'fault_examples': result['faults'][:5],
			'rounds_played': result['rounds_played'],
			'wall_seconds': time.monotonic() - started,
			'cpu_seconds': time.process_time() - cpu,
		}
	except GameBudget:
		return {
			**base,
			'status': 'cpu_timeout',
			'wall_seconds': time.monotonic() - started,
			'cpu_seconds': time.process_time() - cpu,
		}
	except Exception as exc:
		return {
			**base,
			'status': 'error',
			'error': f'{type(exc).__name__}: {exc}',
			'wall_seconds': time.monotonic() - started,
			'cpu_seconds': time.process_time() - cpu,
		}
	finally:
		signal.setitimer(signal.ITIMER_PROF, 0)
		signal.signal(signal.SIGPROF, previous)


def run(stage, scenarios, variants, seeds, out, workers=10, seconds=3600, cpu_limit=180):
	path = Path(out) / f'{stage}.jsonl'
	rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
	done = {(r['scenario'], r['variant'], r['seed']) for r in rows}
	jobs = [
		{'scenario': s, 'variant': v, 'seed': seed, 'cpu_limit': cpu_limit}
		for s in scenarios
		for seed in seeds
		for v in variants
		if (s['id'], v['id'], seed) not in done
	]
	# Complete a shared world/seed block before advancing to another one.
	# This preserves comparable evidence when the phase hits its time limit.
	blocks = {}
	for job in jobs:
		blocks.setdefault((job['scenario']['id'], job['seed']), []).append(job)
	ordered = list(blocks.values())
	random.Random(17).shuffle(ordered)
	jobs = [job for block in ordered for job in block]
	print(f'{stage}: {len(jobs)} games pending; {workers} workers', flush=True)
	if not jobs:
		return rows
	started, last = time.monotonic(), 0
	pool = mp.get_context('spawn').Pool(workers, initializer=initialize, maxtasksperchild=100)
	iterator = pool.imap_unordered(play, jobs, chunksize=1)
	completed = 0
	try:
		with path.open('a') as fh:
			while completed < len(jobs) and time.monotonic() - started < seconds:
				try:
					r = iterator.next(timeout=1)
				except mp.TimeoutError:
					continue
				fh.write(json.dumps(r) + '\n')
				fh.flush()
				rows.append(r)
				completed += 1
				now = time.monotonic()
				if now - last >= 30 or completed == len(jobs):
					print(
						f'{stage}: {completed}/{len(jobs)}; {(now - started) / 60:.1f} min; '
						f'{sum(r["status"] != "ok" for r in rows)} censored/errors',
						flush=True,
					)
					last = now
	finally:
		pool.terminate()
		pool.join()
	return rows
