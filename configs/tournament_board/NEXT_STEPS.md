# Friday handoff — start tournament on GCP

Saved so we can resume without re-deriving the plan.

## Already done
- Pruned chalkboard grid in `tools/tournament_board.py` (`build_cells()`).
- **26,840 cells × 3 seeds = 80,520 runs**.
- `equal+random` uses even **12 / 22** (k per group + 2 random) for board 11/22.
- Parallel runner with `--shard k/n`, resume checkpoints, `--merge` → **exactly 7 final CSVs**.
- GCP auth works for `ak5447@columbia.edu` / project **`amlc-486700`**.
- Existing VMs (TERMINATED): `socks-eval-2`, `socks-eval-6` (`n2-highcpu-16`, `us-east4-a`).

## Friday: begin here

### 1. Create / start VMs
Recommended: **4–6 × `n2-highcpu-16`** in `us-east4-a` (reuse the two socks-eval boxes + create 2–4 new).
Do **not** use GPUs.

Suggested names: `marriage-eval-0` … `marriage-eval-N` (or restart `socks-eval-*` and add new).

### 2. Sync code onto each VM
- Clone/pull `coms4444-marriage-market` at the commit with `tools/tournament_board.py`.
- `uv sync` (or project install) so `uv run` works.
- Confirm: `uv run tools/tournament_board.py --validate-only`

### 3. Launch shards (one per VM)
Example with **4 VMs**:
```bash
# VM 0
uv run tools/tournament_board.py --shard 0/4 --workers 16 --out results/tournament_board

# VM 1
uv run tools/tournament_board.py --shard 1/4 --workers 16 --out results/tournament_board

# VM 2
uv run tools/tournament_board.py --shard 2/4 --workers 16 --out results/tournament_board

# VM 3
uv run tools/tournament_board.py --shard 3/4 --workers 16 --out results/tournament_board
```
Or: `bash tools/tournament_board_gcp.sh K 4`

Checkpoints: `results/tournament_board/checkpoint_shardKof4.jsonl` (resumable if a VM dies).

### 4. After all shards finish — merge once
Collect all `checkpoint_shard*.jsonl` into one directory (if split across VMs, `gcloud compute scp` them together), then:
```bash
uv run tools/tournament_board.py --merge --out results/tournament_board
```
Expect **7 CSVs** in that folder:
1. `1_overall_standings.csv`
2. `2_by_roster.csv`
3. `3_by_bonus.csv`
4. `4_by_rounds.csv`
5. `5_by_dims.csv`
6. `6_by_distribution.csv`
7. `7_raw_runs.csv`

### 5. Stop VMs when done
Stop/delete to avoid idle cost.

## Notes
- Inventory: `configs/tournament_board/CONFIG_LIST.md`, `cells.jsonl`, `pruned_inventory.json`
- Project 1 style reference: `~/Downloads/Results_export/`
- Rough wall time: ~1h on 4×16-vCPU if avg ~2.5s/run; large-N cells can make it longer
- Prompt to use Friday: “begin the VMs and start the tournament”
