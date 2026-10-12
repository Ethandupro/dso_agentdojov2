"""Serve the security-grid page live: every page load rescans the results directory.

Start it once and leave it running; reloading the page picks up new and changed result files. Parsed files are
cached by mtime and size, so only the first load reads everything.

Usage:
    PYTHONPATH=src python util_scripts/serve_results_grid.py runs
    # then open http://localhost:8765
"""

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
from export_results_grid import TEMPLATE, V1_2_VERSION, RecordCache, build_dataset, parse_group, render


@click.command()
@click.argument("logdir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("runs"))
@click.option("--port", default=8765, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--template", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=TEMPLATE)
@click.option("--benchmark-version", "default_version", default=V1_2_VERSION)
@click.option("--group", "groups", multiple=True, help="LABEL=pipeline1,pipeline2 pools runs into one arm. Repeatable.")
def main(logdir: Path, port: int, host: str, template: Path, default_version: str, groups: tuple[str, ...]) -> None:
    cache = RecordCache()
    lock = threading.Lock()
    parsed_groups = [parse_group(group) for group in groups]

    def page() -> bytes:
        with lock:  # one scan at a time; concurrent reloads wait and share the warm cache
            dataset = build_dataset(logdir, default_version, parsed_groups, cache)
        dataset["meta"]["live"] = True
        return render(dataset, template, fragment=False).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    click.echo(f"Reading {logdir} once to warm the cache…")
    page()
    click.echo(f"Serving http://localhost:{port}  (reload the page to pick up new results; Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
