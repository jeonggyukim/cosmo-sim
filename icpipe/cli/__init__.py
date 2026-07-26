"""Command-line entry points for the icpipe library.

Each module exposes a ``main()`` callable that is wired into a
``console_scripts`` entry point in ``pyproject.toml``.  After
``pip install -e .`` the following commands appear on ``$PATH``:

    make-music-conf, compute-pk, compute-pv, plot-ic

The same scripts remain importable as ``python -m icpipe.cli.<name>``.
"""
