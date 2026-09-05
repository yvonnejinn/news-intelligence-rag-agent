# Code delivery and local setup

The GitHub repository stores source code, dependency declarations, small fictional examples, tests, and evidence reports. It does not automatically host a running Python application.

1. Clone this repository using GitHub Desktop or `git clone`.
2. In its directory, create a Python 3.11 environment: `python3 -m venv .venv`.
3. Activate it: `source .venv/bin/activate` (macOS/Linux), or `.venv\Scripts\Activate.ps1` (Windows PowerShell).
4. Follow the README installation and smoke-test commands.
5. Obtain the real dataset/model/API access separately before full experiments.

Use new output directories for each experiment. Keep credentials in environment variables. Only load trusted joblib model bundles. The dependency ranges support installation; capture exact installed versions per experiment before comparing results.

The original Git history is retained. Existing unrelated files are not removed.
