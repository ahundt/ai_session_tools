"""Tests for JWT authentication module."""
import pytest
from auth.jwt import create_token, verify_token

def test_create_token_returns_string():
    token = create_token('user-123')
    assert isinstance(token, str)
    assert len(token) > 0

def test_verify_valid_token():
    token = create_token('user-42')
    payload = verify_token(token)
    assert payload is not None
    assert payload['sub'] == 'user-42'

def test_verify_invalid_token_returns_none():
    result = verify_token('not-a-real-token')
    assert result is None
