"""Builds or checks `skillfile_evolution/pairs_manifest.json`.

python -m agentdojo.runner.build_manifest           # write the manifest
python -m agentdojo.runner.build_manifest --check   # exit 1 if the committed manifest is out of date
"""

import sys
from pathlib import Path

import click

from agentdojo.runner.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestMismatchError,
    build_manifest,
    check_manifest,
    dumps_manifest,
    manifest_sha256,
)


@click.command()
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--check", is_flag=True, help="Compare the existing manifest with a fresh build instead of writing.")
def main(out: Path, check: bool) -> None:
    if check:
        try:
            check_manifest(out)
        except ManifestMismatchError as e:
            click.echo(str(e), err=True)
            sys.exit(1)
        click.echo(f"{out} is up to date (sha256 {manifest_sha256(out)})")
        return
    manifest = build_manifest()
    out.write_text(dumps_manifest(manifest), encoding="utf-8")
    counts = manifest["counts"]
    click.echo(
        f"wrote {out}: {counts['pairs']} pairs ({counts['train_pairs']} train, {counts['test_pairs']} test), "
        f"injection tasks {counts['injection_tasks']}, sha256 {manifest_sha256(out)}"
    )


if __name__ == "__main__":
    main()
