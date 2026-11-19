#!/usr/bin/env python3
"""Start the portable queue as a persistent Linux user service.

Example: python code/start_background.py --name flowers-nov19 -- --source ...
The arguments after -- are passed directly to run_experiments.py.
"""
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="flowers-nov19")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("Use a simple unique service name")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        parser.error("Pass run_experiments.py arguments after --")
    name = args.name + ".service"
    state = subprocess.run(["systemctl", "--user", "is-active", "--quiet", name])
    if state.returncode == 0:
        raise RuntimeError("This service is already running; do not launch another queue")
    executable = [sys.executable, "-u", str(Path(__file__).resolve().with_name("run_experiments.py")), *arguments]
    def quote(value):
        # systemd ExecStart is not a shell. Quote spaces and escape specifier/env expansion.
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'
    unit = ("[Unit]\nDescription=DirectFisheye coordinated selection study\nAfter=network-online.target\n"
            "StartLimitIntervalSec=0\n\n[Service]\nType=simple\nExecStart=" + " ".join(map(quote, executable)) +
            "\nRestart=on-failure\nRestartSec=60\nKillMode=control-group\nTimeoutStopSec=60\n"
            "Environment=PYTHONUNBUFFERED=1\n\n[Install]\nWantedBy=default.target\n")
    folder = Path.home() / ".config/systemd/user"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if path.exists() and path.read_text() != unit:
        raise ValueError("Existing different service will not be overwritten; choose another name")
    path.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", name], check=True)
    print("Started", name)
    subprocess.run(["loginctl", "show-user", os.environ["USER"], "-p", "Linger"], check=True)
    print("For logout survival the line above should read Linger=yes. Follow local administrator policy.")
    print("Monitor: journalctl --user -u", name, "-f")


if __name__ == "__main__":
    main()
