"""Render the targeted sweeps as an offline HTML report and CSV exports.

.venv/bin/python -m players.player_10.targeted_report
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from players.player_10 import grid_report as charts
from players.player_10 import grid_search as grid

plt, np = charts.plt, charts.np


def interval(rows, reference, scenarios=None):
	by_seed = defaultdict(list)
	for r in rows:
		if scenarios is None or r['scenario'] in scenarios:
			base = reference[(r['scenario'], r['seed'])]
			by_seed[r['seed']].append(r['positive_rate'] - base['positive_rate'])
	differences = [statistics.fmean(ds) for ds in by_seed.values()]
	mean = statistics.fmean(differences)
	half = (
		float(charts.student_t.ppf(0.975, len(differences) - 1))
		* statistics.stdev(differences)
		/ math.sqrt(len(differences))
		if len(differences) > 1
		else 0
	)
	return {'mean': mean, 'low': mean - half, 'high': mean + half, 'seeds': len(differences)}


def build(out):
	manifest = json.loads((out / 'experiment.json').read_text())
	selected = json.loads((out / 'selection.json').read_text())
	cfg = manifest['config']
	variants = {v['id']: v for v in selected['variants']}
	labels = {v: variants[v]['label'] for v in variants}
	cases = {s['id']: s for s in manifest['coverage_scenarios'] + manifest['validation_scenarios']}
	rows = [json.loads(line) for line in (out / 'games.jsonl').read_text().splitlines()]
	assert len(rows) == len({(r['phase'], r['scenario'], r['seed'], r['variant']) for r in rows})
	policies = ['defaults', 'previous_winner', 'previous_safer', *selected['finalists']]
	validation = {
		v: [r for r in rows if r['phase'] == 'validation' and r['variant'] == v] for v in policies
	}
	baseline = {(r['scenario'], r['seed']): r for r in validation['defaults']}
	previous = {(r['scenario'], r['seed']): r for r in validation['previous_winner']}
	stats = {
		v: {
			**grid.summarize(rs),
			'vs_defaults': interval(rs, baseline),
			'vs_previous': interval(rs, previous),
		}
		for v, rs in validation.items()
	}
	for rs in validation.values():
		assert len(rs) == len(manifest['validation_scenarios']) * len(cfg['validation_seeds'])
	figdir = out / 'figures'
	figdir.mkdir(exist_ok=True)
	plt.rcParams.update(
		{
			'font.family': 'DejaVu Sans',
			'font.size': 10,
			'svg.fonttype': 'none',
			'svg.hashsalt': 'player10-targeted',
			'figure.facecolor': 'white',
		}
	)
	colors = [charts.GRAY, charts.BLUE, '#6386a3', charts.GREEN, charts.ORANGE]
	plots = {}

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
	for ax, metric, title in zip(
		axes,
		['positive_rate', 'normalized_happiness'],
		['Positive happiness · fresh worlds and seeds', 'Happiness relative to bonus'],
		strict=True,
	):
		values = [stats[v][metric] * 100 for v in policies]
		ax.barh(range(len(policies)), values, color=colors, height=0.62)
		ax.set_yticks(range(len(policies)), [labels[v] for v in policies])
		ax.invert_yaxis()
		for i, value in enumerate(values):
			ax.text(value + 0.5, i, f'{value:.1f}%', va='center')
		ax.set_xlim(min(0, min(values)), max(values) + 10)
		ax.set_xlabel(
			'Player 10 positive rate (%)'
			if metric == 'positive_rate'
			else 'Mean happiness / bonus (%)'
		)
		ax.set_title(title, loc='left', pad=14)
		ax.spines[['top', 'right', 'left']].set_visible(False)
		ax.grid(axis='x', alpha=0.18)
		ax.set_axisbelow(True)
	plots['validation'] = charts.plot_svg(fig, figdir / 'validation.svg')

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
	refinement = {r['variant']: r for r in selected['refinement_ranking']}
	for ax, metric, title in zip(
		axes,
		['positive_rate', 'bad_rate'],
		['Positive happiness (%)', 'Nonpositive marriages / all Player 10 agents (%)'],
		strict=True,
	):
		matrix = np.zeros((len(cfg['threshold_scales']), len(cfg['decay_exponents'])))
		for v in variants.values():
			if v['id'].startswith('t'):
				p = v['parameters']
				i = cfg['threshold_scales'].index(p['threshold_scale'])
				j = cfg['decay_exponents'].index(p['decay_exponent'])
				matrix[i, j] = refinement[v['id']][metric] * 100
		image = ax.imshow(
			matrix, cmap='Blues' if metric == 'positive_rate' else 'Oranges', aspect='auto'
		)
		ax.set_xticks(range(len(cfg['decay_exponents'])), cfg['decay_exponents'])
		ax.set_yticks(range(len(cfg['threshold_scales'])), cfg['threshold_scales'])
		ax.set_xlabel('Decay exponent')
		ax.set_ylabel('Threshold scale')
		ax.set_title(title, loc='left', pad=14, fontsize=11)
		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[1]):
				color = 'white' if image.norm(matrix[i, j]) > 0.65 else '#182a35'
				ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color=color)
		fig.colorbar(image, ax=ax, shrink=0.75)
	plots['thresholds'] = charts.plot_svg(fig, figdir / 'threshold_refinement.svg')

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
	coverage = [r for r in rows if r['phase'] == 'coverage']
	for ax, field, title in zip(
		axes,
		['distribution_name', 'monotonic_mode'],
		['Coverage screen · distribution', 'Coverage screen · monotonic traits'],
		strict=True,
	):
		values = cfg['distributions'] if field == 'distribution_name' else cfg['monotonic_modes']
		for i, v in enumerate(policies[:3]):
			means = [
				statistics.fmean(
					r['positive_rate']
					for r in coverage
					if r['variant'] == v and cases[r['scenario']][field] == value
				)
				* 100
				for value in values
			]
			ax.bar(
				np.arange(len(values)) + (i - 1) * 0.24,
				means,
				width=0.24,
				color=colors[i],
				label=labels[v],
			)
		ax.set_xticks(
			range(len(values)),
			[s.replace('_', '\n') for s in values],
			fontsize=8 if field == 'distribution_name' else 10,
		)
		ax.set_title(title, loc='left', pad=14)
		charts.style_axes(ax)
		ax.set_ylim(0, 100)
	axes[1].legend(frameon=False, loc='lower left', fontsize=9)
	plots['coverage'] = charts.plot_svg(fig, figdir / 'coverage.svg')

	fig, ax = plt.subplots(figsize=(12, 4.1), layout='constrained')
	values = cfg['distributions']
	matrix = np.array(
		[
			[
				interval(
					validation[v],
					baseline,
					{
						s['id']
						for s in manifest['validation_scenarios']
						if s['distribution_name'] == name
					},
				)['mean']
				* 100
				for name in values
			]
			for v in policies[1:]
		]
	)
	image = ax.imshow(matrix, cmap='RdBu', vmin=-20, vmax=20, aspect='auto')
	ax.set_yticks(range(len(policies) - 1), [labels[v] for v in policies[1:]])
	ax.set_xticks(range(len(values)), [s.replace('_', '\n') for s in values])
	ax.set_title(
		'Fresh-seed positive-rate change versus defaults, by distribution', loc='left', pad=16
	)
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			ax.text(
				j,
				i,
				f'{matrix[i, j]:+.1f}',
				ha='center',
				va='center',
				color='white' if abs(matrix[i, j]) > 13 else '#182a35',
			)
	fig.colorbar(image, ax=ax, shrink=0.8, label='Percentage points')
	plots['distribution_gain'] = charts.plot_svg(fig, figdir / 'distribution_gain.svg')

	chosen = selected['finalists'][0]
	fig, ax = plt.subplots(figsize=(10, 3.7), layout='constrained')
	matrix = np.array(
		[
			[
				interval(
					validation[chosen],
					baseline,
					{
						s['id']
						for s in manifest['validation_scenarios']
						if s['dims'] == dims and s['monotonic_mode'] == mode
					},
				)['mean']
				* 100
				for dims in cfg['dims']
			]
			for mode in cfg['monotonic_modes']
		]
	)
	image = ax.imshow(matrix, cmap='RdBu', vmin=-20, vmax=20, aspect='auto')
	ax.set_xticks(range(len(cfg['dims'])), cfg['dims'])
	ax.set_yticks(range(len(cfg['monotonic_modes'])), cfg['monotonic_modes'])
	ax.set_xlabel('Dimensions')
	ax.set_ylabel('Monotonic traits')
	ax.set_title(
		f'Training-selected finalist {chosen}: validation gain versus defaults', loc='left', pad=14
	)
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			ax.text(
				j,
				i,
				f'{matrix[i, j]:+.1f}',
				ha='center',
				va='center',
				color='white' if abs(matrix[i, j]) > 13 else '#182a35',
			)
	fig.colorbar(image, ax=ax, shrink=0.8, label='Percentage points')
	plots['dimensions'] = charts.plot_svg(fig, figdir / 'dimension_gain.svg')

	# Export machine-readable results and all world parameters.
	flat = []
	for r in sorted(rows, key=lambda r: (r['phase'], r['scenario'], r['seed'], r['variant'])):
		case = cases[r['scenario']]
		flat.append(
			{
				**{k: r[k] for k in ['phase', 'scenario', 'seed', 'variant', 'simulation_id']},
				**{
					k: case[k]
					for k in [
						'bonus',
						'rounds',
						'dims',
						'monotonic',
						'monotonic_mode',
						'distribution_name',
						'roster_family',
						'roster',
						'people',
					]
				},
				**{
					k: r[k]
					for k in [
						*grid.METRICS,
						'our_faults',
						'faults',
						'ours',
						'positive',
						'married',
						'bad',
					]
				},
			}
		)
	charts.write_csv(out / 'games.csv', flat)
	charts.write_csv(out / 'coverage_ranking.csv', selected['coverage_ranking'])
	charts.write_csv(
		out / 'refinement_ranking.csv',
		[
			{'rank': i + 1, **variants[r['variant']]['parameters'], **r}
			for i, r in enumerate(selected['refinement_ranking'])
		],
	)
	charts.write_csv(
		out / 'validation_summary.csv',
		[
			{
				'variant': v,
				**{k: val for k, val in stats[v].items() if not isinstance(val, dict)},
				**{f'vs_defaults_{k}': val for k, val in stats[v]['vs_defaults'].items()},
				**{f'vs_previous_{k}': val for k, val in stats[v]['vs_previous'].items()},
			}
			for v in policies
		],
	)
	unique = len({r['simulation_id'] for r in rows})
	faults = sum(r['faults'] for r in {r['simulation_id']: r for r in rows}.values())
	best = max(policies, key=lambda v: grid.rank(stats[v]))
	summary = {
		'rows': len(rows),
		'unique_games': unique,
		'faults': faults,
		'training_selected': chosen,
		'best_observed_validation': best,
		'finalist_parameters': {v: variants[v]['parameters'] for v in selected['finalists']},
		'validation': stats,
	}
	(out / 'report_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
	esc = html.escape
	table = ''
	for v in policies:
		r = stats[v]
		d, previous_delta = r['vs_defaults'], r['vs_previous']
		table += f'<tr><th scope="row">{esc(labels[v])}</th><td>{100 * r["positive_rate"]:.2f}%</td><td>{100 * d["mean"]:+.2f} pp</td><td>[{100 * d["low"]:+.2f}, {100 * d["high"]:+.2f}]</td><td>{100 * previous_delta["mean"]:+.2f} pp</td><td>{100 * r["bad_rate"]:.2f}%</td><td>{100 * r["normalized_happiness"]:.2f}%</td></tr>'
	parameter_rows = ''.join(
		f'<tr><th scope="row"><code>{esc(k)}</code></th>'
		+ ''.join(
			f'<td>{esc(charts.display(variants[v]["parameters"][k]))}</td>'
			for v in selected['finalists']
		)
		+ '</tr>'
		for k in variants[chosen]['parameters']
	)
	coverage_table = ''.join(
		f'<tr><th>{esc(labels[r["variant"]])}</th><td>{100 * r["positive_rate"]:.2f}%</td><td>{100 * r["bad_rate"]:.2f}%</td></tr>'
		for r in selected['coverage_ranking']
	)

	def fig(key, caption):
		return f'<figure role="group" aria-label="{esc(caption)}">{plots[key]}<figcaption>{esc(caption)}</figcaption></figure>'

	c = stats[chosen]
	new = stats[best]
	refinement_base = variants[selected['refinement_base']]['label']
	design = cfg['validation_design_seed']
	threshold_claim = (
		'does not clearly exceed the previous winner'
		if c['vs_previous']['low'] <= 0
		else 'improves on the previous winner within this exploratory validation'
	)
	body = f"""
<header><div class="eyebrow">Player 10 · targeted follow-up experiments</div><h1>Broader coverage. Smaller sweeps.</h1>
<p class="muted">Distribution and monotonicity coverage, followed by a focused threshold search and independent validation. The original mixed-distribution grid remains available separately.</p>
<div class="meta"><div><strong>126</strong>coverage environments</div><div><strong>16</strong>threshold combinations</div><div><strong>{unique:,}</strong>unique games</div><div><strong>3 + 5</strong>training + fresh seeds</div></div></header>
<nav aria-label="Report sections"><a href="#results">Results</a><a href="#thresholds">Threshold refinement</a><a href="#coverage">Coverage</a><a href="#method">Method</a><a href="../player10_grid/report.html">Original grid report</a></nav>
<section id="results"><h2>What changed with broader evaluation conditions?</h2>
<ul><li>The training-selected finalist, <strong>{esc(labels[chosen])}</strong>, achieves <strong>{100 * c['positive_rate']:.2f}%</strong> positive happiness on fresh environments and seeds, versus <strong>{100 * stats['defaults']['positive_rate']:.2f}%</strong> for defaults and <strong>{100 * stats['previous_winner']['positive_rate']:.2f}%</strong> for our previous winner.</li>
<li>Its gain over defaults is <strong>{100 * c['vs_defaults']['mean']:+.2f} percentage points</strong>, with an exploratory paired 95% interval [{100 * c['vs_defaults']['low']:+.2f}, {100 * c['vs_defaults']['high']:+.2f}]. It {threshold_claim}; that comparison's interval is [{100 * c['vs_previous']['low']:+.2f}, {100 * c['vs_previous']['high']:+.2f}] points.</li>
<li>The highest observed validation score is <strong>{100 * new['positive_rate']:.2f}%</strong> for {esc(labels[best])}. This is a descriptive validation ranking; the primary finalist was selected using training results before validation.</li>
<li>The training-selected finalist has <strong>{100 * c['bad_rate']:.2f}%</strong> nonpositive marriages, versus {100 * stats['defaults']['bad_rate']:.2f}% under defaults. All marriage and positive rates use all Player 10 agents as the denominator, including singles.</li>
<li>These percentages come from a broader, differently weighted sample than the first grid. Compare strategies within this report rather than treating the lower overall rates as a direct before/after decline.</li></ul>
{fig('validation', 'Validation: five fresh seeds on 126 newly assigned environments, with equal weight per environment/seed pair.')}
<div class="table-scroll"><table><thead><tr><th>Frozen policy</th><th>Positive</th><th>Δ vs defaults</th><th>95% Δ interval (pp)</th><th>Δ vs previous winner</th><th>Bad marriage</th><th>Happiness / bonus</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="muted" style="margin-top:12px">Intervals are paired Student-t intervals over five seed-level averages (four degrees of freedom). They are exploratory, with few independent seed clusters. Positive happiness is the primary objective; mean happiness divided by bonus breaks training ties. Defaults were not changed.</p>
<h3>Frozen finalist settings</h3><div class="table-scroll"><table class="params"><thead><tr><th>Parameter</th>{''.join(f'<th>{esc(labels[v])}</th>' for v in selected['finalists'])}</tr></thead><tbody>{parameter_rows}</tbody></table></div>
{fig('distribution_gain', 'Paired gains on validation environments, split by distribution. Color saturates at ±20 percentage points; labels retain actual values.')}
{fig('dimensions', 'The training-selected finalist versus defaults, averaged within each dimension/monotonicity cell over seven distributions and five fresh seeds.')}</section>
<section id="thresholds"><h2>Sweep 2 · threshold refinement</h2><p>Scales {esc(str(cfg['threshold_scales']))} × exponents {esc(str(cfg['decay_exponents']))}, holding the other settings at the coverage screen's winner: <strong>{esc(refinement_base)}</strong>. Forty environments were selected to cover every factor value, independently of game outcomes. All 16 settings and the three references replay the same three training seeds.</p>
{fig('thresholds', 'Training results on the 40-case refinement subset. At scale zero, all exponent choices have identical behavior and reuse the same games.')}
<p>The two distinct threshold finalists were frozen before running validation. The chart is a coarse grid over the listed choices, not a continuous optimization.</p></section>
<section id="coverage"><h2>Sweep 1 · distribution and monotonicity coverage</h2><p>All seven distributions × six dimension counts × three monotonicity modes: 126 structural cases. Bonus, round budget, and opponent family are assigned across these cases so all 100 bonus/round/roster-family combinations appear at least once. This is a sampled design, not their full Cartesian product.</p>
<table><thead><tr><th>Reference policy</th><th>Training positive rate</th><th>Training bad-marriage rate</th></tr></thead><tbody>{coverage_table}</tbody></table>
{fig('coverage', 'Coverage-screen results on three training seeds. Cross-category differences are descriptive because other factors vary; policy comparisons within each sampled environment are paired.')}</section>
<section id="method"><h2>Method and scope</h2><ul>
<li><strong>Bonuses:</strong> {esc(str(cfg['bonuses']))}. <strong>Rounds:</strong> {esc(str(cfg['rounds']))}. <strong>Dimensions:</strong> {esc(str(cfg['dims']))}.</li>
<li><strong>Monotonicity:</strong> none (0), half (floor(d/2)), all (d). At one dimension, half equals none.</li>
<li><strong>Distributions:</strong> uniform, normal, mixed; Group 9's U-shaped PMF; Group 10's 90%-zero/10%-ten PMF and its reversed 10%/90% PMF; Group 2's skewed-high PMF. Exact probability tables are recorded for every case.</li>
<li><strong>Opponent families:</strong> all ten groups × 2 (20 people); all groups plus random × 2 (22); half Player 10 and half one opponent (20, rotating through groups 1–9 and random); 90% random and 10% Player 10 (20); all Player 10 self-play (20).</li>
<li><strong>Pairing:</strong> identical profiles, ideals, seat shuffles, and Python/NumPy seeds within each case. All Player 10 agents use the same swept policy. World/seed pairs receive equal weight; self-play does not receive extra weight for having more Player 10 agents.</li>
<li><strong>Data separation:</strong> training seeds {esc(str(cfg['training_seeds']))}; validation seeds {esc(str(cfg['validation_seeds']))}. Validation uses design seed {design} to reassign environments independently of coverage and refinement. The 40-case refinement subset is chosen without inspecting outcomes.</li>
<li><strong>Selection:</strong> coverage chooses the fixed non-threshold settings; refinement chooses two threshold finalists. The validation scores are not used to tune or apply another policy.</li>
<li><strong>Scope:</strong> 20–22 players only. This does not establish performance at 110, 330, or 1,100 players, and does not cross every environment factor. Distribution labels describe full-dimension PMFs, not arbitrary mixtures of PMFs between dimensions.</li>
<li><strong>Unchanged implementation:</strong> existing candidate selection and stale singles-count behavior are preserved. This follow-up keeps round-based decay. No submission or GitHub changes were made.</li>
<li><strong>Simulation quality:</strong> {len(rows):,} measurement rows from {unique:,} unique simulations; {faults:,} player faults. Equivalent zero-threshold exponents, irrelevant monotonic exaggerations, and repeated reference games reuse exact results.</li></ul>
<h3>Reproduce or resume</h3><pre><code>.venv/bin/python -m players.player_10.targeted_sweeps --jobs 10
.venv/bin/python -m players.player_10.targeted_report</code></pre>
<p><a href="experiment.json">Exact cases and source hashes</a> · <a href="selection.json">Frozen selections</a> · <a href="coverage_ranking.csv">Coverage ranking</a> · <a href="refinement_ranking.csv">Threshold ranking</a> · <a href="validation_summary.csv">Validation summary</a> · <a href="games.csv">All measurements</a></p></section>
<footer>All figures are embedded and work offline. Supporting CSV and JSON files sit beside this report.</footer>
"""
	head = (Path(__file__).parent / 'grid_report_template.html').read_text().split('</head>', 1)[0]
	head = head.replace(
		'<title>Player 10 · behavior grid search</title>',
		'<title>Player 10 · targeted sweeps</title>',
	)
	(out / 'report.html').write_text(head + '</head><body><main>' + body + '</main></body></html>')
	print(json.dumps(summary, indent=2))
	print('HTML:', out / 'report.html')


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--out', type=Path, default=grid.ROOT / 'results/player10_targeted')
	args = parser.parse_args()
	build(args.out)


if __name__ == '__main__':
	main()
