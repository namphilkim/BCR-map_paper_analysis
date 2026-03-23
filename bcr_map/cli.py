"""bcr-map CLI."""

from __future__ import annotations

import sys
from typing import Optional

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="bcr-map train | extract",
)


def _run_train(extra: tuple[str, ...]) -> None:
    sys.argv = ["main.py", *extra]
    from bcr_map.train import cli_main

    cli_main()


def _run_extract(extra: tuple[str, ...]) -> None:
    sys.argv = ["extract_vit_patch_embeddings.py", *extra]
    from extract_vit_patch_embeddings import main as extract_main

    extract_main()


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Train (LightningCLI). Help: bcr-map train -- --help",
)
def train(
    ctx: typer.Context,
) -> None:
    _run_train(tuple(ctx.args))


@app.command(
    "extract",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Extract .h5. Help: bcr-map extract -- --help",
)
def extract_cmd(
    ctx: typer.Context,
) -> None:
    _run_extract(tuple(ctx.args))


def main(argv: Optional[list[str]] = None) -> None:
    if argv is not None:
        sys.argv = argv
    app()


if __name__ == "__main__":
    main()
