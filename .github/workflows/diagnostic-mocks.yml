name: Diagnostic mock tests

on:
  push:
    branches:
      - diagnostico/duracoes-workflow

permissions:
  contents: read

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install test dependencies
        run: python -m pip install pytest fastapi python-multipart pillow
      - name: Run diagnostic mock tests only
        run: python -m pytest -q tests/test_diagnostics.py
