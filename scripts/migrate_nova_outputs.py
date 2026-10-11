#!/usr/bin/env python3
"""Safely merge legacy experiment outputs into reorganized directories.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def destination_for(name):
    models = ('deeplabv3plus', 'upernet', 'linknet', 'manet', 'fpn')
    for model in models:
        if name == model or not name.startswith(model + '_'):
            continue
        suffix = name[len(model) + 1:]
        if suffix == 'baseline':
            category = 'baseline'
        elif 'loss_bce' in suffix or 'loss_' in suffix:
            category = 'loss_weights'
        elif suffix.startswith('lr'):
            category = 'learning_rate'
        else:
            return None
        return Path(model) / category / name
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=Path('outputs'))
    p.add_argument('--apply', action='store_true', help='Copy missing files (default: dry-run)')
    args = p.parse_args()
    root = args.outputs.resolve()
    if not root.is_dir():
        p.error(f'Output directory not found: {root}')
    planned = copied = identical = conflicts = 0
    for old in sorted(root.iterdir()):
        if not old.is_dir() or old.is_symlink():
            continue
        relative = destination_for(old.name)
        if relative is None:
            continue
        new = root / relative
        print(f'\n{old.relative_to(root)}  ->  {relative}')
        for source in sorted(old.rglob('*')):
            if source.is_symlink():
                print(f'  SKIP SYMLINK {source.relative_to(old)}')
                continue
            if not source.is_file():
                continue
            relative_file = source.relative_to(old)
            target = new / relative_file
            if target.is_symlink() or any(x.is_symlink() for x in target.parents if x != root and root in x.parents):
                print(f'  CONFLICT unsafe symlink target: {relative_file}')
                conflicts += 1
                continue
            if target.exists():
                if not target.is_file():
                    print(f'  CONFLICT non-file target: {relative_file}')
                    conflicts += 1
                elif digest(source) == digest(target):
                    print(f'  SAME {relative_file}')
                    identical += 1
                else:
                    print(f'  CONFLICT different content: {relative_file}')
                    conflicts += 1
                continue
            planned += 1
            print(f'  {"COPY" if args.apply else "WOULD COPY"} {relative_file}')
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation ensures no overwrite even if another process creates target.
                try:
                    with source.open('rb') as inp, target.open('xb') as out:
                        shutil.copyfileobj(inp, out, 1024 * 1024)
                    if digest(source) != digest(target):
                        print(f'  ERROR hash mismatch: {relative_file}')
                        conflicts += 1
                    else:
                        copied += 1
                except FileExistsError:
                    print(f'  CONFLICT target appeared during copy: {relative_file}')
                    conflicts += 1
    print(f'\nSummary: {planned} missing files, {identical} identical, {conflicts} conflicts, {copied} verified copies.')
    print('Original folders were not modified or deleted.')
    if not args.apply:
        print('Dry-run only. Add --apply to copy missing files.')


if __name__ == '__main__':
    main()
