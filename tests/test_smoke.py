"""Smoke test to verify the test suite runs."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))


def test_negotiation_arena_importable():
    from negotiationarena.game_objects.resource import Resources

    r = Resources({"X": 10, "ZUP": 50})
    assert r.resource_dict == {"X": 10, "ZUP": 50}
