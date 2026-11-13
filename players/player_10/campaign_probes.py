"""Time larger markets before allocating campaign games."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from players.player_10 import campaign_engine as engine
from players.player_10.targeted_sweeps import DEFAULT_CONFIG, make_scenarios, references


def main():
	for name in (
		'OPENBLAS_NUM_THREADS',
		'OMP_NUM_THREADS',
		'MKL_NUM_THREADS',
		'VECLIB_MAXIMUM_THREADS',
	):
		os.environ[name] = '1'
	out = Path('results/player10_campaign')
	out.mkdir(exist_ok=True, parents=True)
	started = datetime.now(UTC)
	(out / 'campaign_clock.json').write_text(
		json.dumps(
			{
				'simulation_started_utc': started.isoformat(),
				'hard_deadline_utc': (started + timedelta(hours=5)).isoformat(),
				'validation_start_utc': (started + timedelta(hours=3, minutes=30)).isoformat(),
			},
			indent=2,
		)
	)
	cfg = json.loads(DEFAULT_CONFIG.read_text())
	base = make_scenarios(cfg, 40261112, 'probe')
	scenarios = []
	for n in (22, 110, 330, 1100):
		for dims, rounds, bonus in ((24, 500, 10), (24, 500, 150), (2, 5, 100)):
			s = next(dict(s) for s in base if s['dims'] == dims)
			s.update(
				id=f'probe_n{n}_d{dims}_r{rounds}_b{bonus}',
				dims=dims,
				monotonic=dims // 2,
				monotonic_mode='half',
				rounds=rounds,
				bonus=bonus,
				people=n,
				seats=[str(i) for i in range(1, 11)] + ['r'],
			)
			s['seats'] *= n // 11
			# Use the exact mixed preset when changing the dimensions.
			s['distribution'] = 'mixed'
			s['distribution_name'] = 'mixed'
			s['roster_family'] = 'all_groups_random'
			scenarios.append(s)
	(out / 'probe_design.json').write_text(
		json.dumps({'scenarios': scenarios, 'variants': references()}, indent=2)
	)
	engine.run(
		'probes', scenarios, references()[:2], [21001], out, workers=4, seconds=1500, cpu_limit=120
	)


if __name__ == '__main__':
	main()
