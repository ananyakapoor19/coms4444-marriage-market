"""Build a self-contained HTML report and CSVs from a completed grid search.

    .venv/bin/python -m players.player_10.grid_report

Plot generation uses the environment's matplotlib, NumPy, and SciPy installations.
The resulting HTML has no network dependencies and embeds its plots and data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault(
	'MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / 'results/.matplotlib')
)
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parents[2] / 'results/.cache'))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from players.player_10.grid_search import METRICS, ROOT, summarize  # noqa: E402

BLUE, ORANGE, GREEN, GRAY = '#245e94', '#c26b28', '#287962', '#9aa5ae'
PARAM_LABELS = {
	'threshold_scale': 'Threshold scale',
	'decay_exponent': 'Decay exponent',
	'decay_by_singles': 'Decay basis',
	'monotonic_exaggeration': 'Monotonic exaggeration',
	'exploration_duration': 'Exploration duration',
	'fresh_candidate_fraction': 'Fresh candidate fraction',
}


def display(value):
	if isinstance(value, bool):
		return 'Singles' if value else 'Rounds'
	if value == 'bonus_scaled':
		return 'int(bonus / 75)'
	return str(value)


def escape(value):
	return html.escape(str(value))


def percent(value):
	return f'{value * 100:.1f}%'


def write_csv(path: Path, rows: list[dict]):
	with path.open('w', newline='') as fh:
		writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def plot_svg(fig, path):
	fig.savefig(path, format='svg', bbox_inches='tight', metadata={'Date': None})
	fig.savefig(path.with_suffix('.png'), dpi=150, bbox_inches='tight')
	plt.close(fig)
	svg = path.read_text()
	svg = svg[svg.index('<svg') :]
	# SVG IDs share the HTML document namespace when figures are embedded.
	for identifier in set(re.findall(r'id="([^"]+)"', svg)):
		prefixed = f'{path.stem}-{identifier}'
		svg = svg.replace(f'id="{identifier}"', f'id="{prefixed}"')
		svg = svg.replace(f'url(#{identifier})', f'url(#{prefixed})')
		svg = svg.replace(f'href="#{identifier}"', f'href="#{prefixed}"')
	return svg


def style_axes(ax, ylabel='Positive happiness (%)'):
	ax.set_ylabel(ylabel)
	ax.spines[['top', 'right']].set_visible(False)
	ax.spines[['left', 'bottom']].set_color('#c8ced3')
	ax.grid(axis='y', alpha=0.18)
	ax.set_axisbelow(True)


def aggregate_by(rows, key):
	groups = defaultdict(list)
	for row in rows:
		groups[key(row)].append(row)
	return {k: summarize(rs) for k, rs in groups.items()}


def build(out: Path):
	manifest = json.loads((out / 'experiment.json').read_text())
	cfg, markets, variants = manifest['config'], manifest['scenarios'], manifest['variants']
	selection = json.loads((out / 'selection.json').read_text())
	validation = json.loads((out / 'validation.json').read_text())
	lookup = {v['id']: v for v in variants}
	scenarios = {s['id']: s for s in markets}
	rows = [json.loads(line) for line in (out / 'games.jsonl').read_text().splitlines()]
	keys = {(r['phase'], r['scenario'], r['seed'], r['variant']) for r in rows}
	assert len(keys) == len(rows), 'Duplicate checkpoint rows'
	grid = [r for r in rows if r['phase'] == 'grid']
	assert len(grid) == len(variants) * len(markets) * len(cfg['training_seeds']), 'Incomplete grid'
	store = {(r['phase'], r['scenario'], r['seed'], r['variant']): r for r in rows}
	baseline, winner = selection['baseline'], selection['global_winner']
	policies = ['baseline', *selection['global_top3'], 'scenario_selected']
	labels = {'baseline': 'Current defaults', 'scenario_selected': 'Scenario-selected settings'}
	labels.update({v: f'{v} · grid #{i + 1}' for i, v in enumerate(selection['global_top3'])})
	policy_rows = {}
	for policy in policies:
		policy_rows[policy] = []
		for s in markets:
			v = (
				baseline
				if policy == 'baseline'
				else (
					selection['scenario_winners'][s['id']]
					if policy == 'scenario_selected'
					else policy
				)
			)
			for seed in cfg['validation_seeds']:
				policy_rows[policy].append(store[('validation', s['id'], seed, v)])

	# With five validation seed clusters, use a Student-t interval rather than
	# the runner's quick normal approximation. Pair worlds before averaging.
	for policy in policies:
		by_seed = defaultdict(list)
		for row in policy_rows[policy]:
			base = store[('validation', row['scenario'], row['seed'], baseline)]
			by_seed[row['seed']].append(row['positive_rate'] - base['positive_rate'])
		differences = [statistics.fmean(values) for values in by_seed.values()]
		mean = statistics.fmean(differences)
		half = (
			float(student_t.ppf(0.975, len(differences) - 1))
			* statistics.stdev(differences)
			/ math.sqrt(len(differences))
			if len(differences) > 1
			else 0
		)
		validation[policy]['positive_delta_vs_baseline'] = {
			'mean': mean,
			'low': mean - half,
			'high': mean + half,
			'n_seeds': len(differences),
		}
		validation[policy]['interval_method'] = 'paired Student-t over seed-level means'

	plt.rcParams.update(
		{
			'font.family': 'DejaVu Sans',
			'font.size': 10,
			'axes.titlesize': 12,
			'axes.labelsize': 10,
			'figure.facecolor': 'white',
			'axes.facecolor': 'white',
			'svg.fonttype': 'none',
			'svg.hashsalt': 'player10-grid-report',
		}
	)
	figures = out / 'figures'
	figures.mkdir(exist_ok=True)
	plots = {}
	fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
	for ax, metric, title, multiplier in zip(
		axes,
		['positive_rate', 'normalized_happiness'],
		['Positive happiness on fresh seeds', 'Happiness relative to bonus on fresh seeds'],
		[100, 100],
		strict=True,
	):
		values = [validation[p][metric] * multiplier for p in policies]
		colors = [
			GRAY if p == 'baseline' else GREEN if p == 'scenario_selected' else BLUE
			for p in policies
		]
		ax.barh(range(len(policies)), values, color=colors, height=0.62)
		ax.set_yticks(range(len(policies)), [labels[p] for p in policies])
		ax.invert_yaxis()
		for i, value in enumerate(values):
			ax.text(value + 0.6, i, f'{value:.1f}%', va='center', fontsize=10)
		ax.set_xlim(min(0, min(values) - 5), max(values) + 13)
		ax.set_xlabel(
			'Player 10 positive rate (%)'
			if metric == 'positive_rate'
			else 'Mean happiness / bonus (%)'
		)
		ax.set_title(title, loc='left', pad=16)
		ax.spines[['top', 'right', 'left']].set_visible(False)
		ax.grid(axis='x', alpha=0.18)
		ax.set_axisbelow(True)
	plots['validation'] = plot_svg(fig, figures / 'validation.svg')

	fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
	sensitivity = []
	for ax, (param, values) in zip(axes.flat, cfg['parameters'].items(), strict=True):
		means = []
		for value in values:
			relevant = [r for r in grid if lookup[r['variant']]['parameters'][param] == value]
			stats = summarize(relevant)
			means.append(stats['positive_rate'] * 100)
			sensitivity.append({'parameter': param, 'value': display(value), **stats})
		ax.bar(range(len(values)), means, color=BLUE, width=0.55)
		ax.set_xticks(range(len(values)), [display(v) for v in values])
		ax.set_ylim(0, max(means) * 1.18)
		for i, value in enumerate(means):
			ax.text(i, value + 0.7, f'{value:.1f}%', ha='center')
		ax.set_title(PARAM_LABELS[param], loc='left', pad=12)
		style_axes(ax)
	plots['sensitivity'] = plot_svg(fig, figures / 'parameter_effects.svg')
	write_csv(out / 'parameter_effects.csv', sensitivity)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
	interaction_specs = [
		(
			cfg['parameters']['threshold_scale'],
			cfg['parameters']['decay_by_singles'],
			lambda r: lookup[r['variant']]['parameters']['threshold_scale'],
			lambda r: lookup[r['variant']]['parameters']['decay_by_singles'],
			'Threshold scale',
			'Decay basis',
			'Decay basis × threshold scale',
		),
		(
			cfg['bonuses'],
			cfg['parameters']['monotonic_exaggeration'],
			lambda r: scenarios[r['scenario']]['bonus'],
			lambda r: lookup[r['variant']]['parameters']['monotonic_exaggeration'],
			'Bonus',
			'Monotonic exaggeration',
			'Exaggeration × bonus',
		),
	]
	for ax, (xs, ys, xvalue, yvalue, xlabel, ylabel, title) in zip(
		axes, interaction_specs, strict=True
	):
		matrix = np.array(
			[
				[
					statistics.fmean(
						r['positive_rate'] for r in grid if xvalue(r) == x and yvalue(r) == y
					)
					* 100
					for x in xs
				]
				for y in ys
			]
		)
		image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
		ax.set_xticks(range(len(xs)), [display(x) for x in xs])
		ax.set_yticks(range(len(ys)), [display(y) for y in ys])
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title, loc='left', pad=14)
		for i in range(len(ys)):
			for j in range(len(xs)):
				ax.text(
					j,
					i,
					f'{matrix[i, j]:.1f}%',
					ha='center',
					va='center',
					color='white' if matrix[i, j] > 55 else '#182a35',
				)
	fig.colorbar(image, ax=list(axes), shrink=0.75, label='Positive happiness (%)')
	plots['interactions'] = plot_svg(fig, figures / 'parameter_interactions.svg')

	def heatmaps(delta):
		fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), layout='constrained')
		for ax, bonus in zip(axes.flat, cfg['bonuses'], strict=False):
			matrix = np.zeros((len(cfg['dims']), len(cfg['rounds'])))
			for i, dims in enumerate(cfg['dims']):
				for j, rounds in enumerate(cfg['rounds']):
					ids = {
						s['id']
						for s in markets
						if s['bonus'] == bonus and s['dims'] == dims and s['rounds'] == rounds
					}
					base = statistics.fmean(
						r['positive_rate'] for r in policy_rows['baseline'] if r['scenario'] in ids
					)
					best = statistics.fmean(
						r['positive_rate'] for r in policy_rows[winner] if r['scenario'] in ids
					)
					matrix[i, j] = (best - base if delta else base) * 100
			image = ax.imshow(
				matrix,
				cmap='RdBu' if delta else 'Blues',
				vmin=-30 if delta else 0,
				vmax=30 if delta else 100,
				aspect='auto',
			)
			ax.set_xticks(range(len(cfg['rounds'])), cfg['rounds'])
			ax.set_yticks(range(len(cfg['dims'])), cfg['dims'])
			ax.set_xlabel('Rounds')
			ax.set_ylabel('Dimensions')
			ax.set_title(f'Bonus {bonus}', loc='left', pad=12)
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):
					val = matrix[i, j]
					text = f'{val:+.1f}' if delta else f'{val:.0f}%'
					ax.text(
						j,
						i,
						text,
						ha='center',
						va='center',
						color='white' if abs(val) > (19 if delta else 55) else '#182a35',
					)
		for unused in list(axes.flat)[len(cfg['bonuses']) :]:
			unused.axis('off')
		fig.colorbar(
			image,
			ax=list(axes.flat)[: len(cfg['bonuses'])],
			shrink=0.65,
			label='Change in positive rate (percentage points)' if delta else 'Positive rate (%)',
		)
		return plot_svg(fig, figures / ('context_gain.svg' if delta else 'context_baseline.svg'))

	plots['context_gain'] = heatmaps(True)
	plots['context_baseline'] = heatmaps(False)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
	for i, policy in enumerate(['baseline', winner, 'scenario_selected']):
		color = [GRAY, BLUE, GREEN][i]
		by_roster = aggregate_by(policy_rows[policy], lambda r: scenarios[r['scenario']]['roster'])
		x = np.arange(len(cfg['rosters']))
		axes[0].bar(
			x + (i - 1) * 0.23,
			[by_roster[r['id']]['positive_rate'] * 100 for r in cfg['rosters']],
			width=0.23,
			color=color,
			label=labels[policy],
		)
		by_rounds = aggregate_by(policy_rows[policy], lambda r: scenarios[r['scenario']]['rounds'])
		axes[1].plot(
			cfg['rounds'],
			[by_rounds[t]['positive_rate'] * 100 for t in cfg['rounds']],
			color=color,
			marker=['o', 's', '^'][i],
			label=labels[policy],
		)
	axes[0].set_xticks(
		range(len(cfg['rosters'])),
		[r['label'].replace(' · ', '\n') for r in cfg['rosters']],
	)
	axes[0].set_title('Opponent mix and population', loc='left', pad=14)
	axes[1].set_title('Round budget', loc='left', pad=14)
	axes[1].set_xlabel('Rounds (actual spacing)')
	axes[1].set_xticks(cfg['rounds'])
	for ax in axes:
		style_axes(ax)
		ax.set_ylim(0, 100)
	axes[1].legend(frameon=False, loc='lower right', fontsize=9)
	plots['contexts'] = plot_svg(fig, figures / 'rosters_and_rounds.svg')

	fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
	ranking = selection['grid_ranking']
	ax.scatter(
		[r['normalized_happiness'] * 100 for r in ranking],
		[r['positive_rate'] * 100 for r in ranking],
		color=BLUE,
		alpha=0.55,
		s=30,
	)
	for v, color, marker, label in [
		(baseline, ORANGE, 'D', 'Defaults'),
		(winner, GREEN, '*', f'{winner}: grid winner'),
	]:
		r = next(r for r in ranking if r['variant'] == v)
		x, y = r['normalized_happiness'] * 100, r['positive_rate'] * 100
		ax.scatter([x], [y], color=color, marker=marker, s=130, label=label, zorder=4)
		ax.annotate(
			label,
			(x, y),
			xytext=(10, -18 if v == baseline else 10),
			textcoords='offset points',
			fontsize=10,
		)
	ax.set_xlabel('Mean happiness / bonus (%)')
	style_axes(ax)
	ax.set_title('All 96 behaviors on the tuning seeds', loc='left', pad=15)
	plots['tradeoff'] = plot_svg(fig, figures / 'grid_tradeoff.svg')

	# Deterministic, flat exports in addition to the append-only checkpoint.
	write_csv(
		out / 'grid_ranking.csv',
		[{'rank': i + 1, **lookup[r['variant']]['parameters'], **r} for i, r in enumerate(ranking)],
	)
	write_csv(
		out / 'validation_summary.csv',
		[
			{
				'policy': p,
				**{k: v for k, v in validation[p].items() if not isinstance(v, dict)},
				**{f'delta_{k}': v for k, v in validation[p]['positive_delta_vs_baseline'].items()},
			}
			for p in policies
		],
	)
	flat = []
	for r in sorted(rows, key=lambda r: (r['phase'], r['scenario'], r['seed'], r['variant'])):
		s = scenarios[r['scenario']]
		flat.append(
			{
				**{k: r[k] for k in ['phase', 'scenario', 'seed', 'variant', 'equivalent_to']},
				**{k: s[k] for k in ['bonus', 'rounds', 'dims', 'monotonic', 'roster', 'people']},
				**{
					k: r[k]
					for k in [
						*METRICS,
						'our_faults',
						'faults',
						'rounds_played',
						'mean_married_round',
						'ours',
						'positive',
						'married',
						'bad',
					]
				},
			}
		)
	write_csv(out / 'games.csv', flat)

	# Compact arrays support an offline scenario explorer without a CDN.
	by_cell = defaultdict(list)
	for row in grid:
		by_cell[(row['scenario'], row['variant'])].append(row)
	explorer = {}
	for s in markets:
		tuning = []
		for v in variants:
			stats = summarize(by_cell[(s['id'], v['id'])])
			tuning.append(
				[
					v['id'],
					*[round(stats[m], 6) for m in METRICS],
					stats['our_faults'],
					stats['faults'],
				]
			)
		heldout = []
		for p in policies:
			stats = summarize([r for r in policy_rows[p] if r['scenario'] == s['id']])
			heldout.append([p, *[round(stats[m], 6) for m in METRICS]])
		explorer[s['id']] = {'tuning': tuning, 'heldout': heldout}

	unique_games = len({(r['phase'], r['scenario'], r['seed'], r['equivalent_to']) for r in rows})
	faults = sum(r['faults'] for r in rows if r['variant'] == r['equivalent_to'])
	our_faults = sum(r['our_faults'] for r in rows if r['variant'] == r['equivalent_to'])
	fault_examples = list(dict.fromkeys(f for r in rows for f in r['fault_examples']))[:8]
	w = validation[winner]
	b = validation['baseline']
	delta = w['positive_delta_vs_baseline']
	win_params = lookup[winner]['parameters']
	params_html = ''.join(
		f'<tr><th scope="row"><code>{escape(k)}</code></th><td>{escape(display(v))}</td></tr>'
		for k, v in win_params.items()
	)
	grid_html = ''.join(
		f'<tr><th scope="row">{escape(PARAM_LABELS[k])}</th><td>{escape(", ".join(display(v) for v in values))}</td></tr>'
		for k, values in cfg['parameters'].items()
	)
	rosters_html = ''.join(
		f'<tr><th scope="row">{escape(r["label"])}</th><td>{escape(", ".join(f"{code} × {count}" for code, count in r["players"]))}</td></tr>'
		for r in cfg['rosters']
	)
	validation_html = ''
	for p in policies:
		r, d = validation[p], validation[p]['positive_delta_vs_baseline']
		validation_html += f'<tr><th scope="row">{escape(labels[p])}</th><td>{percent(r["positive_rate"])}</td><td>{100 * d["mean"]:+.2f} pp</td><td>[{100 * d["low"]:+.2f}, {100 * d["high"]:+.2f}]</td><td>{r["mean_happiness"]:.2f}</td><td>{percent(r["normalized_happiness"])}</td><td>{percent(r["bad_rate"])}</td></tr>'
	winners = Counter(selection['scenario_winners'].values())
	most_common = ', '.join(f'{v} ({n} scenarios)' for v, n in winners.most_common(5))
	findings = [
		f'The training-selected global winner, <strong>{winner}</strong>, achieves <strong>{percent(w["positive_rate"])}</strong> positive happiness on fresh seeds, versus <strong>{percent(b["positive_rate"])}</strong> for the current defaults: <strong>{100 * delta["mean"]:+.2f} percentage points</strong>.',
		f'The approximate 95% paired interval for that improvement is [{100 * delta["low"]:+.2f}, {100 * delta["high"]:+.2f}] percentage points. It uses five seed-level averages; this is an exploratory result, not a precise tournament guarantee.',
		f'The positive-rate gain comes with a tradeoff: the fraction of Player 10 agents ending in a nonpositive marriage rises from <strong>{100 * b["bad_rate"]:.2f}%</strong> under defaults to <strong>{100 * w["bad_rate"]:.2f}%</strong> under {winner}. These percentages use all Player 10 agents as the denominator, including singles.',
		f'Selecting a separate setting for each scenario from the tuning seeds yields {percent(validation["scenario_selected"]["positive_rate"])} positive happiness on fresh seeds. This evaluates a scenario-specific selection procedure; it is not a single deployable parameter setting.',
	]
	if len(selection['global_top3']) > 1:
		alternative = selection['global_top3'][1]
		a = validation[alternative]
		findings.append(
			f'The second-ranked tuning candidate, <strong>{alternative}</strong>, achieves {percent(a["positive_rate"])} positive happiness and {100 * a["bad_rate"]:.2f}% nonpositive marriages on fresh seeds. The validation table and scenario explorer let you compare this tradeoff with the training-selected winner.'
		)
	if delta['low'] <= 0 <= delta['high']:
		findings.append(
			'The improvement interval includes zero, so the current validation does not clearly establish an overall gain for the global winner.'
		)
	findings.append(
		'With singles-based decay and threshold scale 1, a full market starts at a threshold equal to the bonus. The estimate cannot exceed the bonus, so Player 10 cannot initiate marriage while the reported singles count stays unchanged. In homogeneous self-play this can prevent any marriages.'
	)
	findings_html = ''.join(f'<li>{f}</li>' for f in findings)
	quality = f'{our_faults:,} Player 10 faults and {faults:,} total player faults across {unique_games:,} unique games.'
	if fault_examples:
		quality += '<pre>' + escape('\n'.join(fault_examples)) + '</pre>'
	data = {
		'config': cfg,
		'scenarios': [{k: v for k, v in s.items() if k != 'seats'} for s in markets],
		'variants': variants,
		'baseline': baseline,
		'winner': winner,
		'labels': labels,
		'cells': explorer,
	}
	data_json = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
	template = (Path(__file__).parent / 'grid_report_template.html').read_text()
	replacements = {
		'FINDINGS': findings_html,
		'WINNER': winner,
		'WINNER_PARAMS': params_html,
		'PARAM_GRID': grid_html,
		'ROSTERS': rosters_html,
		'VALIDATION_TABLE': validation_html,
		'QUALITY': quality,
		'UNIQUE_GAMES': f'{unique_games:,}',
		'GRID_ROWS': f'{len(grid):,}',
		'VALIDATION_ROWS': f'{len(rows) - len(grid):,}',
		'MOST_COMMON': escape(most_common),
		'DATA': data_json,
		**{f'PLOT_{k.upper()}': v for k, v in plots.items()},
	}
	for key, value in replacements.items():
		template = template.replace(f'@@{key}@@', value)
	assert '@@' not in template, 'Unfilled report placeholder'
	(out / 'report.html').write_text(template)
	summary = {
		'unique_games': unique_games,
		'grid_rows': len(grid),
		'validation_rows': len(rows) - len(grid),
		'our_faults': our_faults,
		'total_faults': faults,
		'global_winner': winner,
		'parameters': win_params,
		'heldout': validation,
	}
	(out / 'report_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
	print(json.dumps(summary, indent=2))
	print(f'HTML: {out / "report.html"}')


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--out', type=Path, default=ROOT / 'results/player10_grid')
	args = parser.parse_args()
	build(args.out)


if __name__ == '__main__':
	main()
