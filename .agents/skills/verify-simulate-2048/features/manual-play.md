# Manual play

**Reach**: `uv run python manuals_control.py` (also `make play`) opens a matplotlib window titled `2048 Game`. Arrow keys move tiles, `backspace` restarts, `escape` closes. Each move prints `pressed <key>` and `reward=<float>`, and `terminated!` when no move is left.

**Set up**: none. Each launch starts a new random board with two tiles.

**Drive**: headless, through the same `key_handler` the window registers.

```bash
.agents/skills/verify-simulate-2048/scripts/drive_manual_play.py "$RUN/play" left up right down backspace escape > "$RUN/play.log" 2>&1
echo "exit=$?"; cat "$RUN/play/boards.txt"
```

Keys are matplotlib key names: `left`, `up`, `right`, `down`, `backspace`, `escape`. The script runs in the foreground and exits on its own; there is no PID to record.

**Proof**:

- `play.log` has one `pressed <key>` line per key and a `reward=` line per arrow key.
- `boards.txt` shows each board after each key. An arrow key slides tiles toward that side, merges equal neighbours (reward equals the sum of merged tiles), and adds one new tile. `backspace` shows a fresh two-tile board. `escape` shows `closed=True`.
- `play/NN-<key>.png` renders the same board as the matching `boards.txt` entry (no PNG after `escape`). Open one and compare.

**Gotchas**:

- `UserWarning: FigureCanvasAgg is non-interactive` is expected under the Agg backend.
- Headless driving proves the key handling, game logic, and rendering. It does not prove that the interactive window opens under the macOS backend. If your change touches window creation or the backend, run `make play` yourself and ask the user to confirm the window.
- An arrow key that cannot move anything still prints `reward=0.00` and leaves the board unchanged: no new tile is added.
- Reaching `terminated!` needs a full game; there is no seed flag or fixture board. Drive many random keys if you must show it.
