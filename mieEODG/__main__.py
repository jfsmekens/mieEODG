"""
Command line entry point for mieEODG.

Mirrors ``python -m rfmwrapper configure``: shows where the ARIA refractive
index database is being resolved from, and records a location in the user
config file.

Usage
-----
    python -m mieEODG configure
    python -m mieEODG configure --aria-dir /path/to/ARIA
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import refindex
from .refindex import (_CONFIG_FILE, _DEFAULT_ARIA_DIR, _cfg_get,
                       get_aria_dir, write_config)


def _source() -> str:
    """Name the layer that ARIA is currently resolving from."""
    if os.environ.get('MIEODG_ARIA_ROOT'):
        return 'MIEODG_ARIA_ROOT'
    if _cfg_get('aria_dir'):
        return f'aria_dir in {_CONFIG_FILE}'
    try:
        resolved = get_aria_dir()
    except FileNotFoundError:
        return 'unresolved'
    if resolved == (Path(refindex.__file__).parent / 'data' / 'ARIA').resolve():
        return 'bundled with package'
    if resolved == _DEFAULT_ARIA_DIR.resolve():
        return '~/ARIA'
    return 'legacy RFM tree'


def _show() -> None:
    """Print the current ARIA configuration."""
    print("\nmieEODG — current configuration")
    print("=" * 60)
    print(f"Config file : {_CONFIG_FILE} "
          f"({'exists' if _CONFIG_FILE.exists() else 'not found'})\n")
    try:
        aria = str(get_aria_dir())
        n_ri = sum(1 for _ in Path(aria).rglob('*.ri'))
        status = f'{n_ri} .ri files'
    except FileNotFoundError as exc:
        aria = 'NOT FOUND'
        status = ''
        print(exc)
    print(f"{'Path':<12} {'Value':<42} {'Source'}")
    print("-" * 78)
    print(f"{'aria_dir':<12} {aria:<42} {_source()}")
    if status:
        print(f"{'':<12} {status}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog='python -m mieEODG')
    sub = parser.add_subparsers(dest='command')

    cfg = sub.add_parser('configure', help='show or set data paths')
    cfg.add_argument('--aria-dir', metavar='PATH',
                     help='record the ARIA database root in the user config file')

    args = parser.parse_args(argv)

    if args.command != 'configure':
        parser.print_help()
        return 1

    if args.aria_dir:
        path = Path(os.path.expanduser(args.aria_dir)).resolve()
        if not path.is_dir():
            print(f"error: not a directory: {path}", file=sys.stderr)
            return 1
        write_config(aria_dir=path)
        print(f"Recorded aria_dir = {path} in {_CONFIG_FILE}")

    _show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
