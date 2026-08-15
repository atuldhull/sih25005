"""Tests for the LLM provider chain. No network calls - the pool and
config logic are what need proving; live cloud behavior is exercised
manually once keys are pasted into keys.json.

Run:  venv\\Scripts\\python test_providers.py
"""
import llm_providers
from llm_providers import KeyPool


def main():
    pool = KeyPool(["k1", "k2", "k3"])
    assert pool.size() == 3
    assert pool.usable_keys() == ["k1", "k2", "k3"]
    pool.cooldown("k1", 60)
    assert pool.usable_keys() == ["k2", "k3"]
    pool.cooldown("k2", 60)
    pool.cooldown("k3", 60)
    assert pool.usable_keys() == []
    print("PASS  key pool: rotation and cooldown exhaustion")

    pool2 = KeyPool(["PASTE_GEMINI_KEY_1", "", "real-key"])
    assert pool2.size() == 1, "placeholder and empty keys must be ignored"
    print("PASS  placeholder/empty keys ignored")

    # with no keys configured, the cloud chain must return instantly
    import time
    t0 = time.perf_counter()
    text, label = llm_providers.try_cloud("system", "user")
    elapsed = time.perf_counter() - t0
    if llm_providers.status()["gemini_keys"] == 0:
        assert text is None and label is None
        assert elapsed < 1.0, f"no-keys path took {elapsed:.2f}s - must be instant"
        print(f"PASS  keyless cloud chain returns (None, None) in {elapsed*1000:.0f}ms")
    else:
        print(f"INFO  live keys configured -> cloud answered via {label}: "
              f"{(text or '')[:60]}...")

    s = llm_providers.status()
    assert "gemini_keys" in s and "gemini_model" in s
    print(f"PASS  status(): {s}")


if __name__ == "__main__":
    main()
