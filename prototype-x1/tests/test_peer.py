"""Tests for peer node handshake and trust pre-authorization."""
import os
import pathlib
import time

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from brain.trust import TrustRegistry, Tier
from brain.peer import build_handshake, verify_handshake, generate_token


def test_peer_preauth():
    t = TrustRegistry.__new__(TrustRegistry)
    t._friends = {}
    t.path = pathlib.Path("/tmp/test_trust_peer.json")
    assert t.get_tier("backup_001") == Tier.FRIEND, "backup_001 should be Friend tier"
    assert t.is_peer("backup_001"), "backup_001 should be recognized as peer"
    assert t.get_tier("stranger") == Tier.UNKNOWN
    assert "pre-authorized Peer" in t.describe("backup_001")


def test_handshake_valid():
    secret = os.urandom(32)
    hs = build_handshake("backup_001", secret)
    ok, reason = verify_handshake(hs, secret)
    assert ok, f"valid handshake should pass: {reason}"


def test_handshake_expired():
    secret = os.urandom(32)
    hs = build_handshake("backup_001", secret)
    old_hs = dict(hs)
    old_hs["ts"] = time.time() - 60  # 60s ago, outside 30s window
    old_hs["token"] = generate_token("backup_001", secret, old_hs["ts"])
    ok, reason = verify_handshake(old_hs, secret)
    assert not ok, "expired token should fail"
    assert "expired" in reason


def test_handshake_tampered():
    secret = os.urandom(32)
    hs = build_handshake("backup_001", secret)
    bad_hs = dict(hs)
    bad_hs["token"] = "deadbeef"
    ok, reason = verify_handshake(bad_hs, secret)
    assert not ok, "tampered token should fail"
    assert "mismatch" in reason


def test_stranger_not_peer():
    t = TrustRegistry.__new__(TrustRegistry)
    t._friends = {}
    t.path = pathlib.Path("/tmp/test_trust_peer.json")
    assert not t.is_peer("stranger")
    assert t.get_tier("stranger") == Tier.UNKNOWN

