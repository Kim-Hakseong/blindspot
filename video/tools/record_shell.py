"""Records a real shell session over a pty (adapted from Backstop / Standing Order).

Not a typing animation: keys go into a live bash on a pty, the shell echoes
them, the commands really run (against AWS where they say so), and every output
chunk is stored with its real timestamp. render_terminal.py replays it exactly.

    uv run python video/tools/record_shell.py reproduce
    uv run python video/tools/record_shell.py agent       # starts a real cloud run (~$0.09)
    uv run python video/tools/record_shell.py rules
    uv run python video/tools/record_shell.py trace
"""

from __future__ import annotations

import json
import os
import pathlib
import pty
import select
import subprocess
import sys
import time

TOOLS = pathlib.Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
OUT = REPO / ".cache" / "video"
SESSION = sys.argv[1]

COMMANDS = {
    # The hook's failing cell, recomputed from scratch on this machine.
    "reproduce": [
        "uv run blindspot probe --set motion_blur.exposure_ms=15.625 "
        "--set low_light.illuminance_lux=173.44977585656073 "
        "--fix low_light.exposure_ms=15.625 --seed 20260906",
    ],
    # The agent proposes; deterministic code decides; the run starts on AWS.
    "agent": [
        "uv run --group cloud --group agent blindspot cloud-run --dataset val/road100 --budget 0.40 --agent",
    ],
    # The rules that keep a model out of the judgment path, enforced as tests.
    "rules": [
        "uv run pytest -p no:warnings tests/test_no_llm_in_judgment.py tests/test_determinism.py "
        "tests/test_gate.py tests/test_approve.py 2>&1 | tail -1",
    ],
    # The ledger of the agent run recorded above: what it read, what it cost, what was refused.
    "trace": [
        "uv run --group cloud python tools/show_decisions.py --run-id 20261006-125326-d307ab --limit 12",
    ],
}[SESSION]

TYPE_DELAY = 0.04
BETWEEN = 1.6
LIMIT_S = 300


def main():
    master, slave = pty.openpty()
    env = dict(os.environ, PS1="$ ", BASH_SILENCE_DEPRECATION_WARNING="1",
               TERM="xterm-256color", COLUMNS="400", LINES="34",
               AWS_PROFILE="blindspot", AWS_REGION="us-east-1",
               JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION="1")
    env.pop("VIRTUAL_ENV", None)
    proc = subprocess.Popen(["/bin/bash", "--norc", "--noprofile", "-i"],
                            stdin=slave, stdout=slave, stderr=slave,
                            cwd=str(REPO), env=env, close_fds=True)
    os.close(slave)
    events, t0 = [], time.time()

    def drain(timeout=0.05):
        while True:
            r, _, _ = select.select([master], [], [], timeout)
            if not r:
                return
            try:
                data = os.read(master, 65536)
            except OSError:
                return
            if not data:                       # EOF: never spin on it
                return
            events.append([round(time.time() - t0, 3), data.decode("utf-8", "replace")])

    drain(0.8)
    for cmd in COMMANDS:
        for ch in cmd:
            os.write(master, ch.encode())
            drain(TYPE_DELAY)
        time.sleep(0.25)
        os.write(master, b"\n")
        drain(0.3)
        start = len(events)
        while time.time() - t0 < LIMIT_S:      # until bash prints a fresh prompt
            drain(0.3)
            since = "".join(e[1] for e in events[start:])
            if since.endswith("\n$ ") or since.endswith("\r\n$ "):
                break
        time.sleep(BETWEEN)
        drain(0.1)
    os.write(master, b"exit\n")
    drain(1.0)
    proc.wait(timeout=10)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pty_{SESSION}.json").write_text(json.dumps(events))
    print(f"{SESSION}: {len(events)} chunks, {events[-1][0]:.1f}s")


main()
