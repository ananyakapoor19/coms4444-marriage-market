# Marriage Market Tournament — Config List (pruned)

- **Unique cells:** 26840
- **Engine runs (×3 seeds):** 80520
- By slice: {'A_core': 21708, 'A_core_micro': 324, 'B_scale': 168, 'C_exotic': 2880, 'D_extreme': 1760}
- By roster: {'equal': 798, 'equal+random': 494, 'solo+random': 6360, '90random_10group': 3240, 'half-half': 14580, 'half-half_sample': 1368}
- equal+random uses board 10/11 & 20/22 as **12 / 22** (k per group + 2 random; engine requires even N).
- Per-group seat counts: `cells.jsonl` → `players`.

## Outputs
Shards only write `checkpoint_shardKofN.jsonl`. After all shards finish:

```bash
uv run tools/tournament_board.py --merge --out results/tournament_board
```

That produces **exactly 7 CSVs** in `results/tournament_board/`:
1. `1_overall_standings.csv`
2. `2_by_roster.csv`
3. `3_by_bonus.csv`
4. `4_by_rounds.csv`
5. `5_by_dims.csv`
6. `6_by_distribution.csv`
7. `7_raw_runs.csv`
