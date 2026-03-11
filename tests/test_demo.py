#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo harness for ai_session_tools (aise CLI).

Dual purpose:
  Standalone: python tests/test_demo.py --record   # record asciinema + produce GIF/MP4
              python tests/test_demo.py --verify   # verify recording tool calls
  Pytest:     pytest tests/test_demo.py::TestDemoFree    # always safe, no cost

PRIVACY: The demo uses ONLY synthetic session data in /tmp/aise-demo/.
No real ~/.claude session data is ever read or recorded.
The synthetic data contains generic dev conversations with no personal information.

Usage:
  # First time or to regenerate:
  python tests/test_demo.py --record

  # Convert to GIF only (cast already exists):
  python tests/test_demo.py --gif-only

  # Verify the recording captured expected commands:
  python tests/test_demo.py --verify

Dependencies:
  asciinema   brew install asciinema
  agg         brew install agg
  ffmpeg      brew install ffmpeg

Output files (gitignored):
  demo.cast   raw asciinema recording
  demo.gif    animated GIF
  demo.mp4    MP4 video
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

# Strip ANSI escape sequences from subprocess output (Rich may emit them).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-B0-2]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

# ── Constants ──────────────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).parent / "aise-demo"
PROJECTS_DIR = DEMO_DIR / "projects"
OUTPUT_DIR = Path(__file__).parent.parent  # repo root
CAST_FILE = OUTPUT_DIR / "demo.cast"
GIF_FILE  = OUTPUT_DIR / "demo.gif"
MP4_FILE  = OUTPUT_DIR / "demo.mp4"

# Post A (self-improvement loop) — separate output files so --post-a never
# collides with --record outputs.
CAST_FILE_POST_A = OUTPUT_DIR / "post-a.cast"
GIF_FILE_POST_A  = OUTPUT_DIR / "post-a.gif"
MP4_FILE_POST_A  = OUTPUT_DIR / "post-a.mp4"

# Set CLAUDE_CONFIG_DIR so aise uses committed fixture data, never ~/.claude.
# Respect an existing CLAUDE_CONFIG_DIR in the environment so that record()
# can pass a date-shifted tmp dir through to the --run-acts subprocess.
DEMO_ENV = {**os.environ,
            "CLAUDE_CONFIG_DIR": os.environ.get("CLAUDE_CONFIG_DIR", str(DEMO_DIR)),
            "NO_COLOR": "1"}

# Whether to add typing delays and pauses (True when recording, False in tests)
_TIMED = False


def pause(seconds: float) -> None:
    """Sleep only when recording (timing mode enabled)."""
    if _TIMED:
        time.sleep(seconds)


def section(title: str) -> None:
    """Print a visual section divider with a descriptive title (no act numbers)."""
    bar_len = max(68, len(title) + 6)  # always wider than the title + 2-space indent
    bar = "─" * bar_len
    sys.stdout.write(f"\n\n\n\033[90m{bar}\033[0m\n")
    sys.stdout.write(f"\033[1;96m  {title}\033[0m\n")
    sys.stdout.write(f"\033[90m{bar}\033[0m\n")
    sys.stdout.flush()


# ── Synthetic Session Data ─────────────────────────────────────────────────────

# Project directory names follow Claude's encoding:
# /Users/demo/<project> → -Users-demo-<project>
# The demo user is called "demo" so no real username appears in paths.

_S1 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000001"))
_S2 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000002"))
_S3 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000003"))
_S4 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000004"))
_S5 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000005"))
_S6 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000006"))
_S7 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000007"))
_S8 = str(uuid.UUID("cafe0001-cafe-cafe-cafe-000000000008"))

def _msg(session_id: str, role: str, content, timestamp: str,
         cwd: str | None = None, git_branch: str | None = None) -> str:
    """Build a single JSONL record."""
    obj: dict = {
        "sessionId": session_id,
        "type": role,
        "timestamp": timestamp,
        "message": {
            "role": role,
            "content": content,
        },
    }
    if cwd:
        obj["cwd"] = cwd
    if git_branch:
        obj["gitBranch"] = git_branch
    return json.dumps(obj)


def _tool_write(session_id: str, timestamp: str, file_path: str, content: str,
                cwd: str | None = None) -> str:
    """Build an assistant JSONL record containing a Write tool call."""
    _id = hashlib.md5(f"{session_id}{timestamp}{file_path}".encode()).hexdigest()[:8]
    return _msg(
        session_id, "assistant",
        [{"type": "tool_use", "id": f"t-{_id}", "name": "Write",
          "input": {"file_path": file_path, "content": content}}],
        timestamp, cwd,
    )


def _tool_edit(session_id: str, timestamp: str, file_path: str,
               old_string: str, new_string: str, cwd: str | None = None) -> str:
    """Build an assistant JSONL record containing an Edit tool call."""
    _id = hashlib.md5(f"{session_id}{timestamp}{file_path}edit".encode()).hexdigest()[:8]
    return _msg(
        session_id, "assistant",
        [{"type": "tool_use", "id": f"t-{_id}", "name": "Edit",
          "input": {"file_path": file_path,
                    "old_string": old_string, "new_string": new_string}}],
        timestamp, cwd,
    )


def _tool_bash(session_id: str, timestamp: str, command: str,
               cwd: str | None = None) -> str:
    """Build an assistant JSONL record containing a Bash tool call."""
    _id = hashlib.md5(f"{session_id}{timestamp}{command}bash".encode()).hexdigest()[:8]
    return _msg(
        session_id, "assistant",
        [{"type": "tool_use", "id": f"t-{_id}", "name": "Bash",
          "input": {"command": command, "description": command[:60]}}],
        timestamp, cwd,
    )


# ── Project 1: webauth — authentication and API ───────────────────────────────

_WEB_CWD = "/Users/demo/webauth"
_WEB_PROJ = "-Users-demo-webauth"

_WEB_SESSION_1 = [
    _msg(_S1, "user",
         "I need to implement JWT authentication for the web app. "
         "Can you create an auth module with login, logout, and token validation?",
         "2026-02-10T09:00:00.000Z", _WEB_CWD, "main"),
    _msg(_S1, "assistant",
         "I'll create a JWT authentication module with login, logout, and token validation.",
         "2026-02-10T09:00:10.000Z", _WEB_CWD),
    _tool_write(_S1, "2026-02-10T09:00:15.000Z",
                f"{_WEB_CWD}/auth/jwt.py",
                "\"\"\"JWT authentication utilities.\"\"\"\nimport jwt\nimport time\n\nSECRET = 'change-in-production'\nALGORITHM = 'HS256'\n\ndef create_token(user_id: str, expires_in: int = 3600) -> str:\n    payload = {'sub': user_id, 'exp': time.time() + expires_in}\n    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)\n\ndef verify_token(token: str) -> dict | None:\n    try:\n        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])\n    except jwt.ExpiredSignatureError:\n        return None\n    except jwt.InvalidTokenError:\n        return None\n",
                _WEB_CWD),
    _msg(_S1, "user",
         "add a login endpoint that validates credentials and returns a token",
         "2026-02-10T09:05:00.000Z", _WEB_CWD, "main"),
    _tool_write(_S1, "2026-02-10T09:05:30.000Z",
                f"{_WEB_CWD}/auth/routes.py",
                "\"\"\"Authentication route handlers.\"\"\"\nfrom flask import Blueprint, request, jsonify\nfrom .jwt import create_token, verify_token\nfrom .models import User\n\nauth_bp = Blueprint('auth', __name__, url_prefix='/auth')\n\n@auth_bp.route('/login', methods=['POST'])\ndef login():\n    data = request.get_json()\n    user = User.query.filter_by(email=data.get('email')).first()\n    if user and user.check_password(data.get('password')):\n        token = create_token(str(user.id))\n        return jsonify({'token': token, 'expires_in': 3600})\n    return jsonify({'error': 'Invalid credentials'}), 401\n\n@auth_bp.route('/logout', methods=['POST'])\ndef logout():\n    # Token invalidation handled client-side\n    return jsonify({'message': 'Logged out successfully'})\n",
                _WEB_CWD),
    _msg(_S1, "user",
         "add unit tests for the jwt functions",
         "2026-02-10T09:10:00.000Z", _WEB_CWD, "main"),
    _tool_write(_S1, "2026-02-10T09:10:45.000Z",
                f"{_WEB_CWD}/tests/test_auth.py",
                "\"\"\"Tests for JWT authentication module.\"\"\"\nimport pytest\nfrom auth.jwt import create_token, verify_token\n\ndef test_create_token_returns_string():\n    token = create_token('user-123')\n    assert isinstance(token, str)\n    assert len(token) > 0\n\ndef test_verify_valid_token():\n    token = create_token('user-42')\n    payload = verify_token(token)\n    assert payload is not None\n    assert payload['sub'] == 'user-42'\n\ndef test_verify_invalid_token_returns_none():\n    result = verify_token('not-a-real-token')\n    assert result is None\n",
                _WEB_CWD),
    _msg(_S1, "assistant",
         "I've created the test file with three test cases covering token creation, "
         "valid token verification, and invalid token handling.",
         "2026-02-10T09:11:00.000Z", _WEB_CWD),
]

_WEB_SESSION_2 = [
    _msg(_S2, "user",
         "the authentication middleware is returning 403 for valid tokens, "
         "can you investigate and fix the issue",
         "2026-02-20T14:00:00.000Z", _WEB_CWD, "fix/auth-middleware"),
    _msg(_S2, "assistant",
         "I'll investigate the authentication middleware issue. Let me check the token verification logic.",
         "2026-02-20T14:00:10.000Z", _WEB_CWD),
    _tool_bash(_S2, "2026-02-20T14:00:15.000Z",
               "grep -n 'verify_token\\|check_auth' /Users/demo/webauth/middleware.py",
               _WEB_CWD),
    _msg(_S2, "user",
         "actually wait you missed the real bug, the issue is in token expiry "
         "not the route handler. also need to fix the middleware comparison logic",
         "2026-02-20T14:00:25.000Z", _WEB_CWD, "fix/auth-middleware"),
    _msg(_S2, "assistant",
         "Found the issue. The middleware is checking token expiry with the wrong timezone. "
         "The verify_token function uses UTC but the middleware compares against local time.",
         "2026-02-20T14:01:00.000Z", _WEB_CWD),
    _tool_edit(_S2, "2026-02-20T14:01:30.000Z",
               f"{_WEB_CWD}/auth/jwt.py",
               "exp': time.time() + expires_in",
               "exp': time.time() + expires_in  # UTC timestamp",
               _WEB_CWD),
    _tool_edit(_S2, "2026-02-20T14:01:45.000Z",
               f"{_WEB_CWD}/middleware.py",
               "if current_time > payload['exp']:",
               "if time.time() > payload['exp']:  # compare UTC",
               _WEB_CWD),
    _msg(_S2, "user",
         "also update the tests to cover the timezone fix",
         "2026-02-20T14:05:00.000Z", _WEB_CWD, "fix/auth-middleware"),
    _tool_edit(_S2, "2026-02-20T14:05:30.000Z",
               f"{_WEB_CWD}/tests/test_auth.py",
               "def test_verify_invalid_token_returns_none():",
               "def test_expired_token_returns_none():\n    token = create_token('user-1', expires_in=-1)\n    assert verify_token(token) is None\n\ndef test_verify_invalid_token_returns_none():",
               _WEB_CWD),
    _msg(_S2, "assistant",
         "Fixed the timezone issue and added a test for expired tokens. "
         "The authentication middleware should now work correctly.",
         "2026-02-20T14:06:00.000Z", _WEB_CWD),
]

# ── Project 2: datapipe — ETL and processing ─────────────────────────────────

_DATA_CWD = "/Users/demo/datapipe"
_DATA_PROJ = "-Users-demo-datapipe"

_DATA_SESSION_3 = [
    _msg(_S3, "user",
         "I need to build a data processing pipeline that reads CSV files, "
         "validates the schema, transforms records, and writes output to parquet.",
         "2026-02-15T10:00:00.000Z", _DATA_CWD, "main"),
    _msg(_S3, "assistant",
         "I'll build a data processing pipeline with CSV input, schema validation, "
         "transformation, and parquet output.",
         "2026-02-15T10:00:15.000Z", _DATA_CWD),
    _tool_write(_S3, "2026-02-15T10:01:00.000Z",
                f"{_DATA_CWD}/pipeline/reader.py",
                "\"\"\"CSV reader with schema validation.\"\"\"\nimport pandas as pd\nfrom pathlib import Path\n\nREQUIRED_COLUMNS = ['id', 'timestamp', 'value', 'category']\n\ndef read_csv(path: str | Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    missing = set(REQUIRED_COLUMNS) - set(df.columns)\n    if missing:\n        raise ValueError(f'Missing required columns: {missing}')\n    return df\n",
                _DATA_CWD),
    _tool_write(_S3, "2026-02-15T10:02:00.000Z",
                f"{_DATA_CWD}/pipeline/transformer.py",
                "\"\"\"Data transformation functions.\"\"\"\nimport pandas as pd\n\ndef normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy()\n    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)\n    return df\n\ndef filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df['value'].notna() & (df['value'] > 0)]\n\ndef transform(df: pd.DataFrame) -> pd.DataFrame:\n    df = normalize_timestamps(df)\n    df = filter_valid_records(df)\n    df['year'] = df['timestamp'].dt.year\n    df['month'] = df['timestamp'].dt.month\n    return df\n",
                _DATA_CWD),
    _tool_write(_S3, "2026-02-15T10:03:00.000Z",
                f"{_DATA_CWD}/pipeline/writer.py",
                "\"\"\"Parquet writer for processed data.\"\"\"\nimport pandas as pd\nfrom pathlib import Path\n\ndef write_parquet(df: pd.DataFrame, output_path: str | Path, partition_cols: list[str] | None = None) -> None:\n    output_path = Path(output_path)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    if partition_cols:\n        df.to_parquet(output_path, partition_cols=partition_cols, index=False)\n    else:\n        df.to_parquet(output_path, index=False)\n",
                _DATA_CWD),
    _msg(_S3, "user",
         "Add a main pipeline script that ties these together.",
         "2026-02-15T10:10:00.000Z", _DATA_CWD, "main"),
    _tool_write(_S3, "2026-02-15T10:10:30.000Z",
                f"{_DATA_CWD}/run_pipeline.py",
                "#!/usr/bin/env python3\n\"\"\"Main data processing pipeline.\"\"\"\nimport argparse\nfrom pipeline.reader import read_csv\nfrom pipeline.transformer import transform\nfrom pipeline.writer import write_parquet\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description='Process CSV data to parquet')\n    parser.add_argument('input', help='Input CSV file path')\n    parser.add_argument('output', help='Output parquet file path')\n    parser.add_argument('--partition', nargs='+', help='Partition columns')\n    args = parser.parse_args()\n\n    print(f'Reading: {args.input}')\n    df = read_csv(args.input)\n    print(f'Loaded {len(df)} records')\n    df = transform(df)\n    print(f'After transform: {len(df)} valid records')\n    write_parquet(df, args.output, partition_cols=args.partition)\n    print(f'Written to: {args.output}')\n\nif __name__ == '__main__':\n    main()\n",
                _DATA_CWD),
]

_DATA_SESSION_4 = [
    _msg(_S4, "user",
         "the pipeline is failing on records with null category values, "
         "can you add better null handling to the transformer",
         "2026-02-25T11:00:00.000Z", _DATA_CWD, "fix/null-handling"),
    _msg(_S4, "user",
         "You forgot to handle None at all, you missed that step. "
         "filter_valid_records still has the old logic, it regressed from what we had before",
         "2026-02-25T11:00:05.000Z", _DATA_CWD, "fix/null-handling"),
    _msg(_S4, "assistant",
         "I'll add null handling to the transformer to handle missing category values gracefully.",
         "2026-02-25T11:00:10.000Z", _DATA_CWD),
    _tool_edit(_S4, "2026-02-25T11:01:00.000Z",
               f"{_DATA_CWD}/pipeline/transformer.py",
               "def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df['value'].notna() & (df['value'] > 0)]",
               "def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy()\n    df['category'] = df['category'].fillna('unknown')\n    return df[df['value'].notna() & (df['value'] > 0)]",
               _DATA_CWD),
    _tool_write(_S4, "2026-02-25T11:02:00.000Z",
                f"{_DATA_CWD}/tests/test_transformer.py",
                "\"\"\"Tests for data transformation functions.\"\"\"\nimport pytest\nimport pandas as pd\nfrom pipeline.transformer import filter_valid_records, transform\n\ndef test_null_category_filled_with_unknown():\n    df = pd.DataFrame({'id': [1], 'timestamp': ['2026-01-01'], 'value': [10.0], 'category': [None]})\n    result = filter_valid_records(df)\n    assert result['category'].iloc[0] == 'unknown'\n\ndef test_negative_values_filtered():\n    df = pd.DataFrame({'id': [1, 2], 'timestamp': ['2026-01-01', '2026-01-01'],\n                       'value': [-5.0, 10.0], 'category': ['A', 'B']})\n    result = filter_valid_records(df)\n    assert len(result) == 1\n    assert result['value'].iloc[0] == 10.0\n",
                _DATA_CWD),
    _msg(_S4, "user",
         "run the tests with uv run python not python3, the deps aren't installed globally",
         "2026-02-25T11:03:00.000Z", _DATA_CWD, "fix/null-handling"),
    _tool_bash(_S4, "2026-02-25T11:03:05.000Z",
               "uv run python -m pytest tests/test_transformer.py -v",
               _DATA_CWD),
    _msg(_S4, "assistant",
         "Added null handling for category values and tests pass with uv run python.",
         "2026-02-25T11:04:00.000Z", _DATA_CWD),
]

# ── Project 3: mlresearch — model training ────────────────────────────────────

_ML_CWD = "/Users/demo/mlresearch"
_ML_PROJ = "-Users-demo-mlresearch"

_ML_SESSION_5 = [
    _msg(_S5, "user",
         "Set up a training script for a text classification model using scikit-learn.",
         "2026-02-18T09:00:00.000Z", _ML_CWD, "main"),
    _tool_write(_S5, "2026-02-18T09:01:00.000Z",
                f"{_ML_CWD}/train.py",
                "\"\"\"Text classification training script.\"\"\"\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\nimport joblib\n\ndef train(texts: list[str], labels: list[str], output_path: str = 'model.pkl') -> None:\n    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)\n    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))\n    X_train_vec = vec.fit_transform(X_train)\n    X_test_vec = vec.transform(X_test)\n    clf = LogisticRegression(max_iter=1000)\n    clf.fit(X_train_vec, y_train)\n    y_pred = clf.predict(X_test_vec)\n    print(classification_report(y_test, y_pred))\n    joblib.dump({'vectorizer': vec, 'classifier': clf}, output_path)\n    print(f'Saved model to {output_path}')\n",
                _ML_CWD),
    _tool_write(_S5, "2026-02-18T09:02:00.000Z",
                f"{_ML_CWD}/predict.py",
                "\"\"\"Prediction script for text classifier.\"\"\"\nimport joblib\n\ndef load_model(path: str = 'model.pkl') -> dict:\n    return joblib.load(path)\n\ndef predict(model: dict, texts: list[str]) -> list[str]:\n    X = model['vectorizer'].transform(texts)\n    return model['classifier'].predict(X).tolist()\n",
                _ML_CWD),
    _msg(_S5, "user",
         "Add experiment tracking with MLflow.",
         "2026-02-18T09:15:00.000Z", _ML_CWD, "main"),
    _tool_edit(_S5, "2026-02-18T09:16:00.000Z",
               f"{_ML_CWD}/train.py",
               "import joblib",
               "import joblib\nimport mlflow",
               _ML_CWD),
    _tool_edit(_S5, "2026-02-18T09:16:30.000Z",
               f"{_ML_CWD}/train.py",
               "def train(texts: list[str], labels: list[str], output_path: str = 'model.pkl') -> None:",
               "def train(texts: list[str], labels: list[str], output_path: str = 'model.pkl') -> None:\n    mlflow.start_run()",
               _ML_CWD),
]

_ML_SESSION_6 = [
    _msg(_S6, "user",
         "I want to add cross-validation to get more reliable accuracy estimates. "
         "Also add authentication to the model API endpoint.",
         "2026-03-01T10:00:00.000Z", _ML_CWD, "feature/cv"),
    _tool_write(_S6, "2026-03-01T10:01:00.000Z",
                f"{_ML_CWD}/evaluate.py",
                "\"\"\"Model evaluation with cross-validation.\"\"\"\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline\n\ndef cross_validate(texts: list[str], labels: list[str], cv: int = 5) -> dict:\n    pipeline = Pipeline([\n        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),\n        ('clf', LogisticRegression(max_iter=1000)),\n    ])\n    scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring='f1_macro')\n    return {'mean_f1': scores.mean(), 'std_f1': scores.std(), 'cv_scores': scores.tolist()}\n",
                _ML_CWD),
    _tool_write(_S6, "2026-03-01T10:05:00.000Z",
                f"{_ML_CWD}/api/auth.py",
                "\"\"\"API key authentication for model endpoint.\"\"\"\nfrom functools import wraps\nfrom flask import request, jsonify\nimport hmac\nimport os\n\nAPI_KEY = os.environ.get('MODEL_API_KEY', 'dev-key-change-in-prod')\n\ndef require_api_key(f):\n    @wraps(f)\n    def decorated(*args, **kwargs):\n        key = request.headers.get('X-API-Key')\n        if not key or not hmac.compare_digest(key, API_KEY):\n            return jsonify({'error': 'Unauthorized'}), 401\n        return f(*args, **kwargs)\n    return decorated\n",
                _ML_CWD),
    _msg(_S6, "assistant",
         "Added cross-validation evaluation and API key authentication for the model endpoint.",
         "2026-03-01T10:06:00.000Z", _ML_CWD),
]

# ── Sessions S7/S8: after the CLAUDE.md fix — no correction-triggering messages ──
# Dated 2026-02-27 (2 days before max_ts 2026-03-01) so create_dated_demo_dir()
# shifts them to ~3 days ago, placing them inside the --since 7d window but with
# zero corrections, providing the visual before/after contrast for Post A Act 6.

_ML_SESSION_7 = [
    _msg(_S7, "user",
         "Add a configuration file for the model training hyperparameters.",
         "2026-02-27T10:00:00.000Z", _ML_CWD, "feature/config"),
    _msg(_S7, "assistant",
         "I'll create a configuration file for the training hyperparameters.",
         "2026-02-27T10:00:10.000Z", _ML_CWD),
    _tool_write(_S7, "2026-02-27T10:01:00.000Z",
                f"{_ML_CWD}/config.py",
                '"""Training hyperparameter configuration."""\n'
                "LEARNING_RATE = 0.01\nMAX_ITER = 1000\nTEST_SIZE = 0.2\n"
                "CV_FOLDS = 5\nRANDOM_STATE = 42\n",
                _ML_CWD),
    _msg(_S7, "user",
         "Looks good. Add a README section for the training config options.",
         "2026-02-27T10:05:00.000Z", _ML_CWD, "feature/config"),
    _msg(_S7, "assistant",
         "Added a README section explaining the training configuration options.",
         "2026-02-27T10:06:00.000Z", _ML_CWD),
]

_ML_SESSION_8 = [
    _msg(_S8, "user",
         "Run the evaluation pipeline on the test dataset and report results.",
         "2026-02-27T14:00:00.000Z", _ML_CWD, "feature/eval"),
    _msg(_S8, "assistant",
         "Running the evaluation pipeline on the test dataset.",
         "2026-02-27T14:00:10.000Z", _ML_CWD),
    _tool_bash(_S8, "2026-02-27T14:01:00.000Z",
               "uv run python evaluate.py --dataset test.csv --output results/",
               _ML_CWD),
    _msg(_S8, "assistant",
         "Evaluation complete. Results written to results/eval_report.json.",
         "2026-02-27T14:02:00.000Z", _ML_CWD),
]

# ── Data generator ─────────────────────────────────────────────────────────────

def create_synthetic_data() -> None:
    """Create synthetic session data in DEMO_DIR. Idempotent."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = [
        (_WEB_PROJ,  _S1, _WEB_SESSION_1),
        (_WEB_PROJ,  _S2, _WEB_SESSION_2),
        (_DATA_PROJ, _S3, _DATA_SESSION_3),
        (_DATA_PROJ, _S4, _DATA_SESSION_4),
        (_ML_PROJ,   _S5, _ML_SESSION_5),
        (_ML_PROJ,   _S6, _ML_SESSION_6),
        (_ML_PROJ,   _S7, _ML_SESSION_7),  # after fix — no corrections (Post A Act 6)
        (_ML_PROJ,   _S8, _ML_SESSION_8),  # after fix — no corrections (Post A Act 6)
    ]

    for proj_dir, session_id, messages in sessions:
        proj_path = PROJECTS_DIR / proj_dir
        proj_path.mkdir(parents=True, exist_ok=True)
        session_file = proj_path / f"{session_id}.jsonl"
        session_file.write_text("\n".join(messages) + "\n")

    _setup_recovery_files(DEMO_DIR)


def _setup_recovery_files(demo_dir: Path) -> None:
    """Create recovery directory structure so 'aise files search' returns results.

    SessionRecoveryEngine.search() reads session_*/ dirs for files.
    SessionRecoveryEngine.get_versions() reads session_all_versions_*/ for edit count.

    Files with 2+ versions (edits) will pass --min-edits 2:
      jwt.py:         S1 (Write) + S2 (Edit) → 2 versions ✓
      test_auth.py:   S1 (Write) + S2 (Edit) → 2 versions ✓
      transformer.py: S3 (Write) + S4 (Edit) → 2 versions ✓
    """
    recovery = demo_dir / "recovery"

    JWT_V1 = (
        '"""JWT authentication utilities."""\n'
        "import jwt\nimport time\n\n"
        "SECRET = 'change-in-production'\nALGORITHM = 'HS256'\n\n"
        "def create_token(user_id: str, expires_in: int = 3600) -> str:\n"
        "    payload = {'sub': user_id, 'exp': time.time() + expires_in}\n"
        "    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)\n\n"
        "def verify_token(token: str) -> dict | None:\n"
        "    try:\n"
        "        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])\n"
        "    except jwt.ExpiredSignatureError:\n"
        "        return None\n"
        "    except jwt.InvalidTokenError:\n"
        "        return None\n"
    )
    JWT_V2 = JWT_V1.replace(
        "exp': time.time() + expires_in}",
        "exp': time.time() + expires_in}  # UTC timestamp",
    )
    ROUTES_V1 = (
        '"""Authentication route handlers."""\n'
        "from flask import Blueprint, request, jsonify\n"
        "from .jwt import create_token, verify_token\n"
        "from .models import User\n\n"
        "auth_bp = Blueprint('auth', __name__, url_prefix='/auth')\n\n"
        "@auth_bp.route('/login', methods=['POST'])\n"
        "def login():\n"
        "    data = request.get_json()\n"
        "    user = User.query.filter_by(email=data.get('email')).first()\n"
        "    if user and user.check_password(data.get('password')):\n"
        "        token = create_token(str(user.id))\n"
        "        return jsonify({'token': token, 'expires_in': 3600})\n"
        "    return jsonify({'error': 'Invalid credentials'}), 401\n\n"
        "@auth_bp.route('/logout', methods=['POST'])\n"
        "def logout():\n"
        "    return jsonify({'message': 'Logged out successfully'})\n"
    )
    TEST_AUTH_V1 = (
        '"""Tests for JWT authentication module."""\n'
        "import pytest\nfrom auth.jwt import create_token, verify_token\n\n"
        "def test_create_token_returns_string():\n"
        "    token = create_token('user-123')\n"
        "    assert isinstance(token, str)\n"
        "    assert len(token) > 0\n\n"
        "def test_verify_valid_token():\n"
        "    token = create_token('user-42')\n"
        "    payload = verify_token(token)\n"
        "    assert payload is not None\n"
        "    assert payload['sub'] == 'user-42'\n\n"
        "def test_verify_invalid_token_returns_none():\n"
        "    result = verify_token('not-a-real-token')\n"
        "    assert result is None\n"
    )
    TEST_AUTH_V2 = TEST_AUTH_V1.replace(
        "def test_verify_invalid_token_returns_none():",
        "def test_expired_token_returns_none():\n"
        "    token = create_token('user-1', expires_in=-1)\n"
        "    assert verify_token(token) is None\n\n"
        "def test_verify_invalid_token_returns_none():",
    )
    TRANSFORMER_V1 = (
        '"""Data transformation functions."""\n'
        "import pandas as pd\n\n"
        "def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = df.copy()\n"
        "    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)\n"
        "    return df\n\n"
        "def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    return df[df['value'].notna() & (df['value'] > 0)]\n\n"
        "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = normalize_timestamps(df)\n"
        "    df = filter_valid_records(df)\n"
        "    df['year'] = df['timestamp'].dt.year\n"
        "    df['month'] = df['timestamp'].dt.month\n"
        "    return df\n"
    )
    TRANSFORMER_V2 = (
        '"""Data transformation functions."""\n'
        "import pandas as pd\nfrom typing import Optional\n\n"
        "def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = df.copy()\n"
        "    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)\n"
        "    return df\n\n"
        "def filter_valid_records(df: pd.DataFrame, min_value: Optional[float] = None) -> pd.DataFrame:\n"
        "    mask = df['value'].notna()\n"
        "    if min_value is not None:\n"
        "        mask &= df['value'] >= min_value\n"
        "    else:\n"
        "        mask &= df['value'] > 0\n"
        "    return df[mask]\n\n"
        "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = normalize_timestamps(df)\n"
        "    df = filter_valid_records(df)\n"
        "    df['year'] = df['timestamp'].dt.year\n"
        "    df['month'] = df['timestamp'].dt.month\n"
        "    return df\n"
    )

    # (session_id, {filename: content}, version_number)
    # version_number must be unique per filename across sessions to count as separate edits
    _sessions: list[tuple] = [
        (_S1, {"jwt.py": JWT_V1, "routes.py": ROUTES_V1, "test_auth.py": TEST_AUTH_V1}, 1),
        (_S2, {"jwt.py": JWT_V2, "test_auth.py": TEST_AUTH_V2},                         2),
        (_S3, {"transformer.py": TRANSFORMER_V1},                                        1),
        (_S4, {"transformer.py": TRANSFORMER_V2},                                        2),
    ]

    for sid, files, vnum in _sessions:
        # session_*/ dir: final-version files (iterated by search())
        final_dir = recovery / f"session_{sid}"
        final_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (final_dir / name).write_text(content)

        # session_all_versions_*/ dir: versioned files (counted by get_versions())
        av_dir = recovery / f"session_all_versions_{sid}"
        av_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            lines = len(content.splitlines())
            (av_dir / f"{name}_v{vnum:06d}_line_{lines}.txt").write_text(content)


def cleanup_synthetic_data() -> None:
    """Remove synthetic data directory."""
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)


# ── Date-shifted demo fixtures ─────────────────────────────────────────────────

_TS_RE = re.compile(r'"timestamp":\s*"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"')


def create_dated_demo_dir() -> Path:
    """Copy DEMO_DIR to a temp dir with all timestamps shifted so the latest
    session appears 1 day before today.  The demo can then use --since 3d and
    always find sessions regardless of when it is re-recorded.

    The committed fixture files are never modified — caller must delete the
    returned temp dir when done.
    """
    # Find the maximum timestamp across all fixture JSONL files
    max_ts: datetime | None = None
    for jsonl in (DEMO_DIR / "projects").rglob("*.jsonl"):
        for m in _TS_RE.finditer(jsonl.read_text()):
            ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%fZ")
            if max_ts is None or ts > max_ts:
                max_ts = ts
    if max_ts is None:
        raise RuntimeError("No timestamps found in fixture JSONL files")

    # Shift: place the latest session 1 day before today
    delta = (datetime.utcnow() - timedelta(days=1)) - max_ts

    # Copy projects and recovery subtrees to a fresh temp dir
    tmp_dir = Path(tempfile.mkdtemp(prefix="aise-demo-dated-"))
    shutil.copytree(DEMO_DIR / "projects", tmp_dir / "projects")
    shutil.copytree(DEMO_DIR / "recovery",  tmp_dir / "recovery")

    def _shift(m: re.Match) -> str:
        ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%fZ")
        return f'"timestamp": "{(ts + delta).strftime("%Y-%m-%dT%H:%M:%S.000Z")}"'

    for jsonl in (tmp_dir / "projects").rglob("*.jsonl"):
        jsonl.write_text(_TS_RE.sub(_shift, jsonl.read_text()))

    return tmp_dir


# ── Demo recording ─────────────────────────────────────────────────────────────

def _build_banner() -> str:
    """Build the intro banner with ASCII art title, ANSI colors, and GitHub link.

    Design principles:
    - ANSI codes applied OUTSIDE center()/ljust() so visible-char alignment is correct
    - ASCII art uses ansi_shadow figlet font (box-drawing chars match banner borders)
    - Hardcoded art avoids runtime dependency on pyfiglet
    - Two ╠═╣ separators give the tagline its own visual band

    Fact-checked (run aise commands to verify):
    - aise stats:   Sessions/Files/Versions counts — NOT message count
    - aise list:    session table with project and date columns
    - aise files:   edit counts per file across sessions (--min-edits filter)
    - aise corrections: auto-classifies via engine.py DEFAULT_CORRECTION_PATTERNS
    - Sources:      Claude Code (SessionRecoveryEngine), AI Studio (AiStudioSource),
                    Gemini CLI (GeminiCliSource) — per __init__.py + pyproject.toml
    - GitHub URL:   github.com/ahundt/ai_session_tools — from pyproject.toml
    """
    W = 90  # visible chars inside box (between ║ borders)

    BOX   = "\033[36m"   # dim cyan — box outline (╔ ═ ║ ╣ ╚ etc.)
    CYAN  = "\033[96m"   # bright cyan — text, art, commands
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    GRAY  = "\033[90m"
    RST   = "\033[0m"

    # ASCII art for "aise" — ansi_shadow figlet font, letters defined separately so
    # a 3-space gap can be inserted between each one for legibility.
    # Each letter is exactly its natural width (a=8, i=3, s=8, e=8 chars per row).
    # Box-drawing chars (║ ═ ╗ ╚ ╔ █) visually match the banner borders.
    _a = [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"]
    _i = ["██╗",      "██║",      "██║",      "██║",      "██║",      "╚═╝"     ]
    _s = ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"]
    _e = ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"]
    _SP = "   "  # 3-space gap between letters for readability
    # Each row: 8 + 3 + 3 + 3 + 8 + 3 + 8 = 36 visible chars — all identical width
    _ART = [a + _SP + i + _SP + s + _SP + e for a, i, s, e in zip(_a, _i, _s, _e)]

    def top() -> str:
        return f"{BOX}  ╔{'═' * W}╗{RST}"

    def sep() -> str:
        return f"{BOX}  ╠{'═' * W}╣{RST}"

    def bot() -> str:
        return f"{BOX}  ╚{'═' * W}╝{RST}"

    def row(text: str = "", style: str = "") -> str:
        """Left-justified content row."""
        content = (" " + text).ljust(W)
        return f"{BOX}  ║{RST}{style}{content}{RST}{BOX}║{RST}"

    def crow(text: str = "", style: str = "") -> str:
        """Centered content row. Padding computed on plain text width."""
        content = text.center(W)
        return f"{BOX}  ║{RST}{style}{content}{RST}{BOX}║{RST}"

    def art_row(text: str) -> str:
        """ASCII art row — center(W) without rstrip so all rows stay the same width."""
        content = text.center(W)
        return f"{BOX}  ║{RST}{BOLD}{CYAN}{content}{RST}{BOX}║{RST}"

    def cmd_row(cmd: str, desc: str, cmd_width: int = 26) -> str:
        """Two-tone row: bold-cyan command + gray description, padded to W.

        Padding computed on visible chars only — no ANSI codes in the math.
        """
        indent = "    "   # 4 visible chars
        arrow  = "  →  "  # 5 visible chars
        cmd_padded = cmd.ljust(cmd_width)
        # visible between ║: 1 (leading sp) + indent + cmd_width + arrow + desc + padding
        padding = " " * max(0, W - 1 - len(indent) - cmd_width - len(arrow) - len(desc))
        return (
            f"{BOX}  ║{RST}"
            f" {indent}"
            f"{BOLD}{CYAN}{cmd_padded}{RST}"
            f"{GRAY}{arrow}{desc}{RST}"
            f"{padding}"
            f"{BOX}║{RST}"
        )

    lines = [
        "",
        top(),
        row(),
        *[art_row(line) for line in _ART],    # 6-row ASCII art title "aise" (ansi_shadow font)
        row(),
        crow("ai_session_tools — search, recover, and analyze AI sessions", style=BOLD),
        row(),
        crow("github.com/ahundt/ai_session_tools"),
        sep(),
        row(),
        crow("Context compacted? Sessions lost? aise gives your history back.", style=GREEN),
        row(),
        sep(),
        row(),
        row("  Commands shown in this demo:"),
        row(),
        cmd_row("aise stats",                "session, file, and version statistics"),
        row(),
        cmd_row("aise messages search",       "find recent prompts or any past conversation"),
        row(),
        cmd_row("aise list",                  "all sessions across every project"),
        row(),
        cmd_row("aise files search",          "files Claude edited most, sorted by edit count"),
        row(),
        cmd_row("aise messages corrections",  "AI correction patterns, auto-detected"),
        row(),
        cmd_row("aise messages get",          "recover the full content of any session"),
        row(),
        crow("Claude Code: /ar:claude-session-tools  (via autorun: github.com/ahundt/autorun)",
             style=GRAY),
        bot(),
        "",
    ]
    return "\n".join(lines)


BANNER = _build_banner()


def _build_post_a_banner() -> str:
    """Build the Post A intro banner for the self-improvement loop demo."""
    W = 90
    BOX  = "\033[36m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RST  = "\033[0m"

    # Same helper closures as _build_banner() — see that function for design notes.
    def top() -> str:
        return f"{BOX}  ╔{'═' * W}╗{RST}"
    def sep() -> str:
        return f"{BOX}  ╠{'═' * W}╣{RST}"
    def bot() -> str:
        return f"{BOX}  ╚{'═' * W}╝{RST}"
    def row(text: str = "", style: str = "") -> str:
        content = (" " + text).ljust(W)
        return f"{BOX}  ║{RST}{style}{content}{RST}{BOX}║{RST}"
    def crow(text: str = "", style: str = "") -> str:
        content = text.center(W)
        return f"{BOX}  ║{RST}{style}{content}{RST}{BOX}║{RST}"
    def cmd_row(cmd: str, desc: str, cmd_width: int = 26) -> str:
        indent = "    "
        arrow  = "  →  "
        cmd_padded = cmd.ljust(cmd_width)
        padding = " " * max(0, W - 1 - len(indent) - cmd_width - len(arrow) - len(desc))
        return (
            f"{BOX}  ║{RST} {indent}"
            f"{BOLD}{CYAN}{cmd_padded}{RST}"
            f"{GRAY}{arrow}{desc}{RST}"
            f"{padding}{BOX}║{RST}"
        )

    lines = [
        "",
        top(),
        row(),
        crow("aise: the Claude Code self-improvement loop", style=BOLD),
        row(),
        crow("github.com/ahundt/ai_session_tools"),
        sep(),
        row(),
        row("  Commands shown in this demo:"),
        row(),
        cmd_row("aise messages corrections", "AI correction patterns, auto-classified"),
        row(),
        cmd_row("aise messages search",      "search all your messages across sessions"),
        row(),
        cmd_row("aise messages corrections", "verify the loop closed after the fix"),
        row(),
        sep(),
        row(),
        crow("Install: uv tool install git+https://github.com/ahundt/ai_session_tools"),
        crow("Claude Code: /ar:claude-session-tools  (via autorun: github.com/ahundt/autorun)",
             style=GRAY),
        bot(),
        "",
    ]
    return "\n".join(lines)


BANNER_POST_A = _build_post_a_banner()


def _type(text: str, delay: float = 0.04) -> None:
    """Write text to stdout character by character with typing effect."""
    if _TIMED:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


def _run(cmd: str, show_prompt: bool = True) -> subprocess.CompletedProcess:
    """Print a shell prompt + command (with typing effect), then run it via shell."""
    if show_prompt:
        _type("\n\033[1;32m$\033[0m ", delay=0)
        _type(cmd + "\n", delay=0.045)
        pause(0.3)
    # Use shell=True so glob patterns like *.py and quotes work naturally
    result = subprocess.run(
        cmd,
        env=DEMO_ENV,
        shell=True,
        capture_output=False,   # let output flow to terminal (captured by asciinema)
        text=True,
    )
    return result


def get_first_session_id() -> str:
    """Return the session ID of the most recent session in synthetic data."""
    # Session 6 (ml-experiments) has the most recent timestamp (2026-03-01)
    return _S6


def run_demo_acts() -> None:
    """Execute all demo acts in order. Call this inside the asciinema recording."""
    # ── Act 0: Intro banner ────────────────────────────────────────────────────
    # Print the full banner instantly so the very first recording frame shows
    # the complete banner — no per-line animation delay.
    sys.stdout.write("\033[H\033[2J")   # clear screen
    sys.stdout.flush()
    sys.stdout.write(BANNER + "\n")
    sys.stdout.flush()
    pause(12.0)  # hold for viewers to read the richer banner

    # All commands use --provider claude to show only synthetic demo sessions,
    # preventing any real session data from the user's system from appearing.
    PROV = "--provider claude"

    # ── Act 1: aise stats ─────────────────────────────────────────────────────
    section("Statistics — sessions, messages, and files indexed")
    pause(1.5)
    _run(f"aise stats {PROV}")
    pause(5.0)

    # ── Act 2: recent user messages (last 3 days) ────────────────────────────
    # Compute --since dynamically; fixtures were date-shifted by create_dated_demo_dir()
    # so this always catches recent sessions regardless of when the demo is re-recorded.
    since = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    section("Recent prompts — your messages to Claude over the last 3 days")
    pause(1.5)
    _run(f"aise messages search '' --type user --since {since} {PROV}")
    pause(5.0)

    # ── Act 3: aise list ──────────────────────────────────────────────────────
    section("Session list — all sessions across every project")
    pause(1.5)
    _run(f"aise list {PROV}")
    pause(5.0)

    # ── Act 4: messages search --context 1 ───────────────────────────────────
    section("Message search — find any past conversation instantly")
    pause(1.5)
    _run(f"aise messages search authentication --context 1 --limit 3 {PROV}")
    pause(5.0)

    # ── Act 5: files search ────────────────────────────────────────────────────
    section("File history — which files Claude edited most")
    pause(1.5)
    _run(f"aise files search --pattern '*.py' --min-edits 2 {PROV}")
    pause(5.0)

    # ── Act 6: messages corrections ───────────────────────────────────────────
    section("AI corrections — patterns in mistakes, detected automatically")
    pause(1.5)
    _run(f"aise messages corrections {PROV}")
    pause(6.0)

    # ── Act 7: messages get SESSION_ID --limit 10 (context rescue) ────────────
    session_id = get_first_session_id()
    section("Session recovery — restore context from a previous session")
    pause(1.5)
    _run(f"aise messages get {session_id} --limit 10 {PROV}")
    pause(6.0)

    sys.stdout.write(
        "\n\n"
        "\033[1;32m  ══════════════════════════════════════════════════════════════════\033[0m\n"
        "\033[1;32m  ✓  Demo complete — ai_session_tools recovers your AI session history\033[0m\n"
        "\033[1;32m  ══════════════════════════════════════════════════════════════════\033[0m\n"
        "\n"
        "  Install:      uv pip install git+https://github.com/ahundt/ai_session_tools\n"
        "  Claude Code:  /ar:claude-session-tools  (via autorun: https://github.com/ahundt/autorun)\n"
        "\n"
    )
    sys.stdout.flush()
    pause(6.0)


def run_post_a_acts() -> None:
    """Execute Post A acts (self-improvement loop demo). Called inside asciinema."""
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()
    sys.stdout.write(BANNER_POST_A + "\n")
    sys.stdout.flush()
    pause(4.0)

    PROV = "--provider claude"

    # ── Act 1: corrections --since 30d ────────────────────────────────────────
    section("Corrections — 30 days of patterns, auto-classified")
    pause(1.5)
    _run(f"aise messages corrections --since 30d {PROV}")
    pause(6.0)

    # ── Act 2: all user messages --since 30d ──────────────────────────────────
    section("Your messages — everything you wrote across sessions this month")
    pause(1.5)
    _run(f"aise messages search '' --type user --since 30d {PROV}")
    pause(5.0)

    # ── Act 3: regex search with context ──────────────────────────────────────
    section("Unclassified feedback — regex search + context-after")
    pause(1.5)
    _run(
        f"aise messages search 'forgot|missed|wrong' --type user --regex"
        f" --context-after 2 {PROV}"
    )
    pause(5.0)

    # ── Act 4: pipeline — session IDs from corrections into targeted search ────
    # shell=True (inside _run) makes the pipe work correctly on macOS/Linux.
    # {{}} in the f-string produces literal {} which xargs needs for -I{} placeholder.
    section("Pipeline — corrections IDs into targeted session search")
    pause(1.5)
    _run(
        f"aise messages corrections --since 14d --ids-only {PROV}"
        f" | xargs -I{{}} aise messages search 'you forgot'"
        f" --session {{}} --context-after 3 {PROV}"
    )
    pause(6.0)

    # ── Act 5: the CLAUDE.md fix — typed display only, no aise command ─────────
    section("The fix — one line in CLAUDE.md")
    pause(1.5)
    _type("You must always use `uv run python`. Never run `python3` or `python` directly.\n")
    pause(3.0)

    # ── Act 6: corrections --since 7d — verify the loop closed ────────────────
    section("Verification — same command, shorter window, after the fix")
    pause(1.5)
    _run(f"aise messages corrections --since 7d {PROV}")
    # Write a final newline to anchor a PTY event just before the hold pause.
    # Without this, the last cast event is the final byte of aise output, and
    # agg/ffmpeg may drop the tail silence during GIF→MP4 conversion.
    sys.stdout.write("\n")
    sys.stdout.flush()
    pause(8.0)


# ── Recording pipeline ─────────────────────────────────────────────────────────

def record(cast_file: Path = CAST_FILE, *, acts_flag: str = "--run-acts") -> None:
    """Record the demo with asciinema."""
    asciinema = shutil.which("asciinema")
    if not asciinema:
        sys.exit("asciinema not found. Install: brew install asciinema")

    create_synthetic_data()
    print(f"Synthetic data created in {DEMO_DIR}")

    # Copy fixtures to a temp dir with timestamps shifted to be near today,
    # so --since 3d in Act 2 always catches sessions regardless of re-record date.
    # Committed fixture files are never modified; temp dir is cleaned up after.
    dated_dir = create_dated_demo_dir()
    print(f"Date-shifted fixtures in {dated_dir} (auto-cleaned after recording)")

    # Remove existing cast so we always get a fresh recording
    if cast_file.exists():
        cast_file.unlink()
    cast_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording to {cast_file} ...")

    # asciinema v3 API: --command, --window-size COLSxROWS, --capture-env
    # Use asciicast-v2 format for compatibility with agg 1.7.0
    cmd = f"{sys.executable} {__file__} {acts_flag}"
    # Pass dated_dir via CLAUDE_CONFIG_DIR so run_demo_acts() uses shifted timestamps.
    # DEMO_ENV respects this env var (see definition above) so all aise commands
    # inside --run-acts will target the dated tmp dir, not the committed fixtures.
    record_env = {**os.environ, "CLAUDE_CONFIG_DIR": str(dated_dir)}
    try:
        subprocess.run([
            asciinema, "rec", str(cast_file),
            "--command", cmd,
            "--window-size", "160x48",
            "--capture-env", "TERM,COLORTERM,CLAUDE_CONFIG_DIR",
            "--output-format", "asciicast-v2",
            "--quiet",
        ], env=record_env, check=True)
    finally:
        shutil.rmtree(dated_dir, ignore_errors=True)
        print(f"Temp fixtures cleaned up")
    print(f"Recording saved to {cast_file}")


def convert_to_gif(
    cast_file: Path = CAST_FILE,
    gif_file: Path = GIF_FILE,
    *,
    speed: float = 0.75,
) -> None:
    """Convert .cast → .gif using agg.

    Args:
        cast_file: Source .cast file.
        gif_file:  Destination .gif file.
        speed:     Playback speed multiplier passed to agg (default 0.75 = 33% slower
                   than recorded, matching the standard demo pacing). Use 1.0 for
                   real-time playback (e.g. Post A, which has shorter built-in pauses).
    """
    agg = shutil.which("agg")
    if not agg:
        print("[demo] agg not found — GIF skipped")
        print("[demo] Install: brew install agg")
        return
    print(f"Converting to GIF: {gif_file} ...")
    subprocess.run([
        agg, str(cast_file), str(gif_file),
        "--theme", "dracula",
        "--font-size", "16",       # readable at 160-col width; agg reads cols/rows from cast
        "--renderer", "fontdue",   # vector-quality anti-aliased text (vs default bitmap)
        "--speed", str(speed),
        "--idle-time-limit", "10",
        "--last-frame-duration", "5",
        "--quiet",
    ], check=True)
    print(f"GIF saved to {gif_file}")


def convert_to_mp4(gif_file: Path = GIF_FILE, mp4_file: Path = MP4_FILE) -> None:
    """Convert .gif → .mp4 using ffmpeg.

    Encoder priority:
      1. libx264 with -tune animation (universally available, best compression for
         terminal/animation content — CRF 24 typically yields 300-600 kbps).
      2. h264_videotoolbox (Apple Silicon hardware, fast but less efficient).
      3. Bare ffmpeg default (last resort).

    fps=24 in the vf chain produces a constant-frame-rate stream suitable for web
    distribution (Reddit, etc.).  agg at speed=1.0 emits GIF frames at the 10 ms
    GIF minimum resolution (100 fps base), so without the fps filter the output
    would be 100 fps and several times larger than necessary.  24 fps is the cinema
    standard; terminal recordings need no more than 24-30 fps.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[demo] ffmpeg not found — MP4 skipped")
        return
    print(f"Converting to MP4: {mp4_file} ...")

    # fps=24: constant frame rate for web compatibility; scale: ensure even dimensions.
    vf = "fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    # libx264 first: universally available, -tune animation dramatically improves
    # compression for content with large static regions (terminal recordings).
    # CRF 24 = good quality/size balance; preset slow = better compression.
    encoders = [
        ["-c:v", "libx264", "-preset", "slow", "-crf", "24",
         "-tune", "animation", "-pix_fmt", "yuv420p"],
        ["-c:v", "h264_videotoolbox", "-b:v", "2M", "-color_range", "tv"],
        ["-pix_fmt", "yuv420p"],
    ]
    for enc_flags in encoders:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(gif_file), "-movflags", "faststart",
             "-vf", vf, *enc_flags, str(mp4_file)],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"MP4 saved to {mp4_file}")
            return
    print("[demo] All ffmpeg encoders failed — MP4 skipped")


# ── Verification ───────────────────────────────────────────────────────────────

# Checks for the default --record demo. Each entry: (fragment, description).
# Fragment must appear in command OUTPUT (not in typed keystroke stream).
_VERIFY_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    ("Sessions:",        "Act 1: stats shows Sessions: label (table format)"),
    ("accuracy",         "Act 2: recent user messages shows S6 content"),
    ("cafe0001",         "Act 3: list shows synthetic session UUIDs"),
    ("authentication",   "Act 4: message search finds authentication"),
    (".py",              "Act 5: files search shows Python files"),
    ("corrections",      "Act 6: corrections command shows AI correction history"),
    ("cross-validation", "Act 7: session get shows ML session content"),
)

# Checks for the --post-a self-improvement loop demo.
_POST_A_VERIFY_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    ("regression",  "Act 1: regression category present in corrections output"),
    ("skip_step",   "Act 1: skip_step category present in corrections output"),
    ("accuracy",    "Act 2: user message stream shows fixture message text"),
    ("missed",      "Act 3: regex search finds missed/forgot correction patterns across sessions"),
    # Act 4: pipeline search output — fixture message starts with capital 'You forgot'.
    # The typed command is char-by-char (never contiguous in cast); check output instead.
    ("You forgot",  "Act 4: pipeline search output shows uv run python correction"),
    # Act 5: _type() is char-by-char so the rule text is never contiguous in cast events.
    # Check the section header written by section() via sys.stdout.write() instead.
    ("CLAUDE.md",   "Act 5: CLAUDE.md fix section header displayed"),
    ("corrections", "Act 6: verification corrections command produced output"),
)


def verify_recording(
    cast_file: Path = CAST_FILE,
    checks: tuple[tuple[str, str], ...] | None = None,
) -> bool:
    """Parse the asciinema cast to verify expected content appeared.

    Args:
        cast_file: Path to the .cast file to verify.
        checks: Sequence of (fragment, description) pairs. Defaults to
                _VERIFY_CHECKS (the standard demo checks) when None.
    """
    if checks is None:
        checks = _VERIFY_CHECKS
    if not cast_file.exists():
        print(f"Cast file not found: {cast_file}")
        return False

    content = cast_file.read_text()
    passed = 0
    for fragment, description in checks:
        if fragment in content:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description} — '{fragment}' not found in cast")
    print(f"\nVerification: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


# ── pytest-compatible free tests ──────────────────────────────────────────────

class TestDemoFree:
    """No-cost tests — verify committed fixture data and aise commands work correctly.

    Fixtures live in tests/aise-demo/ (committed to the repo, ~32 KB).
    No per-test setup/teardown needed — create_synthetic_data() is idempotent
    and can regenerate fixtures with: python tests/test_demo.py --setup
    """

    def test_synthetic_data_created(self) -> None:
        """Verify synthetic data directory structure is correct."""
        assert PROJECTS_DIR.exists()
        proj_dirs = list(PROJECTS_DIR.iterdir())
        assert len(proj_dirs) == 3, f"Expected 3 projects, got {len(proj_dirs)}"
        for proj in [_WEB_PROJ, _DATA_PROJ, _ML_PROJ]:
            assert (PROJECTS_DIR / proj).exists(), f"Missing project dir: {proj}"

    def test_synthetic_sessions_valid_jsonl(self) -> None:
        """All synthetic session files contain valid JSONL with required fields."""
        for jsonl_file in PROJECTS_DIR.rglob("*.jsonl"):
            for i, line in enumerate(jsonl_file.read_text().splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert "sessionId" in obj, f"{jsonl_file}:{i}: missing sessionId"
                assert "type" in obj, f"{jsonl_file}:{i}: missing type"
                assert "timestamp" in obj, f"{jsonl_file}:{i}: missing timestamp"
                assert "message" in obj, f"{jsonl_file}:{i}: missing message"

    def test_aise_stats_runs(self) -> None:
        """aise stats returns exit code 0 with synthetic data."""
        result = subprocess.run(
            ["aise", "stats", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"aise stats failed: {result.stderr}"

    def test_aise_list_runs(self) -> None:
        """aise list returns exit code 0 and shows synthetic session ID prefixes."""
        result = subprocess.run(
            ["aise", "list", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"aise list failed: {result.stderr}"
        # Default output shows 8-char prefix; use --full-uuid for full UUIDs.
        assert _S1[:8] in result.stdout or _S6[:8] in result.stdout, \
            "Expected synthetic session ID prefix in list output"

    def test_aise_messages_search_authentication(self) -> None:
        """aise messages search finds 'authentication' in synthetic sessions."""
        result = subprocess.run(
            ["aise", "messages", "search", "authentication", "--context", "1",
             "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, \
            f"search failed (rc={result.returncode}):\n{_strip_ansi(result.stderr)}"
        # The word "authentication" appears in sessions S1, S2, S6
        assert "authentication" in _strip_ansi(result.stdout).lower(), \
            "Expected 'authentication' in search results"

    def test_aise_files_search_python(self) -> None:
        """aise files search finds Python files in synthetic sessions."""
        result = subprocess.run(
            ["aise", "files", "search", "--pattern", "*.py", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"files search failed: {result.stderr}"

    def test_aise_messages_inspect_runs(self) -> None:
        """aise messages inspect runs for a known session ID."""
        result = subprocess.run(
            ["aise", "messages", "inspect", _S1, "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"messages inspect failed: {result.stderr}"

    def test_aise_messages_recent_user_messages(self) -> None:
        """aise messages search --type user --since returns recent user messages."""
        result = subprocess.run(
            ["aise", "messages", "search", "", "--type", "user",
             "--since", "2026-02-28", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"messages search --type user failed: {result.stderr}"
        # The ML session (2026-03-01) user message contains these words
        assert "accuracy" in result.stdout or "cross-validation" in result.stdout, \
            "Expected ML session user message content in results"

    def test_aise_messages_corrections_runs(self) -> None:
        """aise messages corrections exits 0 and finds corrections in synthetic data."""
        result = subprocess.run(
            ["aise", "messages", "corrections", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"corrections failed: {result.stderr}"
        # Should find correction records from the added messages
        assert "corrections" in result.stdout.lower() or "regression" in result.stdout, \
            f"Expected correction output, got: {result.stdout[:200]}"

    def test_aise_messages_get_session(self) -> None:
        """aise messages get SESSION_ID returns session messages."""
        result = subprocess.run(
            ["aise", "messages", "get", _S6, "--limit", "5", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"messages get failed: {result.stderr}"
        assert result.stdout.strip(), "Expected message output"

    def test_aise_stats_json_format(self) -> None:
        """aise stats --format json produces valid JSON."""
        result = subprocess.run(
            ["aise", "stats", "--format", "json", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stats json failed: {result.stderr}"
        obj = json.loads(result.stdout)
        assert "total_sessions" in obj, f"Missing total_sessions in: {obj}"

    def test_no_real_user_data_in_synthetic_sessions(self) -> None:
        """Verify synthetic data contains no real usernames or personal paths."""
        for jsonl_file in PROJECTS_DIR.rglob("*.jsonl"):
            content = jsonl_file.read_text()
            # The demo uses "demo" as the username, not any real name
            # Real home dirs would be /Users/<realname> — we check the project dirs
            # only contain "demo" as the user
            assert "/Users/demo/" in content or _WEB_CWD in content or \
                   _DATA_CWD in content or _ML_CWD in content, \
                f"Unexpected path in {jsonl_file}"
            # No actual home directory of the running user should appear
            real_home = str(Path.home())
            assert real_home not in content, \
                f"Real home directory {real_home} found in synthetic data: {jsonl_file}"

    def test_aise_list_project_filter(self) -> None:
        """aise list --project webauth finds sessions in the webauth synthetic project."""
        result = subprocess.run(
            ["aise", "list", "--project", "webauth", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"aise list --project webauth failed: {result.stderr}"
        assert _S1[:8] in result.stdout or _S2[:8] in result.stdout, \
            f"Expected webauth session IDs in output; got:\n{result.stdout}"

    def test_find_not_in_aise_help(self) -> None:
        """aise --help must not list 'find' as an explicit command (hidden alias)."""
        result = subprocess.run(
            ["aise", "--help"], env=DEMO_ENV, capture_output=True, text=True,
        )
        stdout = _strip_ansi(result.stdout)
        assert result.returncode == 0, \
            f"aise --help failed (rc={result.returncode}):\n{_strip_ansi(result.stderr)}"
        assert not re.search(r"[│\s]\s*find\s{2,}", stdout), \
            f"'find' must be hidden alias, not listed in aise --help:\n{stdout}"

    def test_aise_list_full_uuid(self) -> None:
        """aise list --full-uuid shows full 36-char UUIDs for synthetic sessions."""
        result = subprocess.run(
            ["aise", "list", "--full-uuid", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"aise list --full-uuid failed: {result.stderr}"
        assert _S1 in result.stdout or _S6 in result.stdout, \
            f"Expected full UUID in --full-uuid output; got:\n{result.stdout}"

    def test_aise_list_compact_column_header(self) -> None:
        """aise list must not show old 'Summary' column header (was renamed to 'Compact')."""
        result = subprocess.run(
            ["aise", "list", "--provider", "claude"],
            env=DEMO_ENV, capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Column may be suppressed when no sessions have compact summaries, but the old
        # name 'Summary' must never appear. The rename is unit-verified by test_list_spec_compact_not_summary.
        assert "Summary" not in result.stdout, \
            f"Old 'Summary' column header must not appear (renamed to 'Compact'); got:\n{result.stdout}"


# ── Main entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    global _TIMED

    parser = argparse.ArgumentParser(
        description="ai_session_tools demo recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--record",    action="store_true", help="Record cast + convert to GIF/MP4")
    parser.add_argument("--gif-only",  action="store_true", help="Convert existing cast to GIF/MP4 only")
    parser.add_argument("--verify",    action="store_true", help="Verify recording contains expected commands")
    parser.add_argument("--run-acts",  action="store_true", help="[internal] Run demo acts inside asciinema")
    parser.add_argument("--setup",     action="store_true", help="Create synthetic data only")
    parser.add_argument("--cleanup",   action="store_true", help="Remove synthetic data")
    parser.add_argument("--post-a",          action="store_true",
                        help="Record Post A (self-improvement loop) cast + GIF/MP4")
    parser.add_argument("--run-post-a-acts", action="store_true",
                        help="[internal] Run Post A acts inside asciinema subprocess")
    parser.add_argument("--verify-post-a",   action="store_true",
                        help="Verify Post A recording contains expected output")
    args = parser.parse_args()

    if args.run_acts:
        # Called by asciinema — run acts with timing delays
        _TIMED = True
        create_synthetic_data()  # regenerate fixtures if missing (idempotent)
        run_demo_acts()

    elif args.record:
        create_synthetic_data()
        record()
        convert_to_gif()
        convert_to_mp4()
        verify_recording()
        print("\nDone! Files:")
        for f in [CAST_FILE, GIF_FILE, MP4_FILE]:
            if f.exists():
                size_kb = f.stat().st_size // 1024
                print(f"  {f}  ({size_kb} KB)")

    elif args.gif_only:
        convert_to_gif()
        convert_to_mp4()

    elif args.verify:
        ok = verify_recording()
        sys.exit(0 if ok else 1)

    elif args.setup:
        create_synthetic_data()
        print(f"Synthetic data created in {DEMO_DIR}")
        for proj_dir in PROJECTS_DIR.iterdir():
            sessions = list(proj_dir.glob("*.jsonl"))
            print(f"  {proj_dir.name}: {len(sessions)} session(s)")

    elif args.cleanup:
        cleanup_synthetic_data()
        print(f"Removed {DEMO_DIR}")
        print("Note: re-commit tests/aise-demo/ if you removed the fixture directory.")

    elif args.run_post_a_acts:
        # Called by asciinema subprocess — run Post A acts with timing delays
        _TIMED = True
        create_synthetic_data()
        run_post_a_acts()

    elif args.post_a:
        create_synthetic_data()
        record(CAST_FILE_POST_A, acts_flag="--run-post-a-acts")
        convert_to_gif(CAST_FILE_POST_A, GIF_FILE_POST_A, speed=1.0)
        convert_to_mp4(GIF_FILE_POST_A, MP4_FILE_POST_A)
        verify_recording(CAST_FILE_POST_A, checks=_POST_A_VERIFY_CHECKS)
        print("\nDone! Post A files:")
        for f in [CAST_FILE_POST_A, GIF_FILE_POST_A, MP4_FILE_POST_A]:
            if f.exists():
                size_kb = f.stat().st_size // 1024
                print(f"  {f}  ({size_kb} KB)")

    elif args.verify_post_a:
        ok = verify_recording(CAST_FILE_POST_A, checks=_POST_A_VERIFY_CHECKS)
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
