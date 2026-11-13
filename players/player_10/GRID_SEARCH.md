# Player 10 behavior search

Run the complete grid and fresh-seed validation from the repository root:

```sh
.venv/bin/python -m players.player_10.grid_search --jobs 10
.venv/bin/python -m players.player_10.grid_report
```

The HTML report is `results/player10_grid/report.html`. It embeds its plots
and scenario-explorer data and works offline. Supporting CSVs, JSON files,
and SVG/PNG figures are written beside it. Plot generation requires
Matplotlib, NumPy, and SciPy in the Python environment.

The configuration is `configs/grid_search.json` relative to this directory.
It specifies a full Cartesian product of 96 behavior settings and 320
environments. The training run uses seeds 4444–4446. It freezes the top three
global settings and one winner per environment, then validates them against
the current defaults on seeds 9001–9005. The grid contains the current
defaults as a reference setting.

All Player 10 agents in a game use the same swept setting. Other players
retain their implementations. Scores are calculated from the engine's true
profiles and ideals. Settings are applied after normal Player 10
initialization; the sweep does not edit `player.py`.

Variants replay identical seeds and shuffled seats within each environment.
Python and NumPy global random streams are reset before each game. Settings
that resolve to identical behavior parameters for a particular bonus reuse
one simulation. For example, `bonus_scaled` exaggeration means
`int(bonus / 75)`, which equals the explicit zero-bump setting at bonus 10.

Ranking prioritizes positive-happiness rate, then mean happiness divided by
bonus. Environment/seed pairs receive equal weight, irrespective of the
number of Player 10 agents in that environment. This prevents the self-play
roster and large bonuses from dominating the objective.

Completed games are appended to `games.jsonl`; repeating the same command
resumes missing games. Configuration, runner, and simulation source hashes
are recorded in `experiment.json`. The runner refuses to combine results
after those inputs change: choose a new `--out` directory. The report
generator can be edited independently of the simulation experiment.

Important interpretation limits:

- This is a coarse grid over the configured values, not a continuous optimum.
- Only the mixed distribution and `min(2, dims)` monotonic traits are tested.
- Three tuning seeds and five validation seeds give an exploratory comparison.
  The report uses paired Student-t intervals over seed-level means; the runner’s
  preliminary validation JSON uses a quick normal approximation.
- Scenario-specific winners are selected on tuning seeds and evaluated on
  fresh seeds; they are not a single parameter setting.
- Self-play changes all participants' behavior together.
- The existing Player 10 exploration logic is preserved, including its
  inactive `dated` updates and candidate/ID membership check.
- Singles-based decay uses the last publicly reported singles count;
  automatic repeat dates do not provide a fresh count.
- With singles-based decay, scale 1, and an unchanged population, the
  threshold equals the bonus and no Player 10 estimate can exceed it.
  This can stall self-play completely.

`tests/test_player10_grid.py` verifies instance overrides, baseline equivalence,
deterministic replay, interrupted-checkpoint recovery, and equivalent-game reuse.

## Targeted follow-up sweeps

```sh
.venv/bin/python -m players.player_10.targeted_sweeps --jobs 10
.venv/bin/python -m players.player_10.targeted_report
```

These commands write a separate report to `results/player10_targeted/report.html`.
The configuration is `configs/targeted_sweeps.json`. A 126-case coverage screen
crosses dimensions, monotonicity modes, and distributions, while rotating
bonuses, rounds, and five 20–22-player roster families. All 100 combinations
of those latter three factors appear at least once; their full product with
the structural factors is not run.

The screen compares defaults and the two strongest reference policies from
the first grid. Its winner supplies the fixed non-threshold parameters for
a 16-setting scale/exponent search on 40 balanced coverage cases. Two
threshold finalists are frozen before validation against all three references
on new seeds and newly assigned environments. Existing Player 10 defaults
and the original full-grid results remain unchanged.

`tests/test_player10_targeted.py` checks factor coverage, distribution tables,
custom-distribution delivery to the engine, and valid cache equivalences.
