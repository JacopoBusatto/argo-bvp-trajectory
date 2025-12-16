## Setup

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -e .[dev]
pytest
