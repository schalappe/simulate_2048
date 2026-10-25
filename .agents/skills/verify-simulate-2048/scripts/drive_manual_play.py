#!/usr/bin/env -S uv run python
"""
Drive manuals_control.py headlessly: send real key events to its matplotlib window and save a PNG per key.

Usage (from the repo root):
    .agents/skills/verify-simulate-2048/scripts/drive_manual_play.py OUT_DIR left up right down backspace escape

Each key goes through `manuals_control.key_handler`, the same callback the GUI registers. The script prints the
board after every key and writes OUT_DIR/NN-<key>.png plus OUT_DIR/boards.txt.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path.cwd()))

from matplotlib.backend_bases import KeyEvent  # noqa: E402

import manuals_control  # noqa: E402
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight  # noqa: E402
from twentyfortyeight.utils.windows import WindowBoard  # noqa: E402


def main() -> int:
    out_dir = Path(sys.argv[1])
    keys = sys.argv[2:]
    out_dir.mkdir(parents=True, exist_ok=True)

    env = TwentyFortyEight()
    window = WindowBoard(title='2048 Game', size=env.size)
    window.register_key_handler(lambda event: manuals_control.key_handler(env, window, event))
    manuals_control.reset(env, window)

    log = [f'00-start\n{env.observation}\n']
    window.fig.savefig(out_dir / '00-start.png')
    for index, key in enumerate(keys, start=1):
        event = KeyEvent('key_press_event', window.fig.canvas, key=key)
        window.fig.canvas.callbacks.process('key_press_event', event)
        label = f'{index:02d}-{key}'
        log.append(f'{label} closed={window.closed} finished={env.is_finished}\n{env.observation}\n')
        if not window.closed:
            window.fig.savefig(out_dir / f'{label}.png')

    text = '\n'.join(log)
    (out_dir / 'boards.txt').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
