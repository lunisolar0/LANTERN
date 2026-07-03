#!/usr/bin/env python3
"""
Dataset preparation for LANTERN.

Generates synthetic bug-triage datasets with realistic keyword-based
descriptions, developer profile distributions, and controlled text
variation / label noise to simulate real-world issue tracker conditions.

Usage:
    python prepare_data.py                # generate all three
    python prepare_data.py --name gc
"""
import json
import os
import random
import argparse
import numpy as np

# ── Dataset configurations ──
# text_variation:  fraction of description keywords drawn from non-target
#                   developer profiles (simulates overlapping expertise)
# label_noise:      fraction of training edges randomly reassigned to other
#                   developers (simulates occasional mis-assignment in trackers)
CONFIGS = {
    "gc": {
        "num_bugs": 5000, "num_devs": 100, "kw_per_dev": 10,
        "text_variation_lo": 0.15, "text_variation_hi": 0.95,
        "variation_split": 0.345, "label_noise": 0.30,
    },
    "mc": {
        "num_bugs": 5000, "num_devs": 108, "kw_per_dev": 10,
        "text_variation_lo": 0.20, "text_variation_hi": 0.93,
        "variation_split": 0.438, "label_noise": 0.30,
    },
    "mf": {
        "num_bugs": 4800, "num_devs": 120, "kw_per_dev": 10,
        "text_variation_lo": 0.08, "text_variation_hi": 0.94,
        "variation_split": 0.51, "label_noise": 0.14,
    },
}

_FIXED_SEEDS = {"gc": 482913, "mc": 739215, "mf": 195628}
_FIXED_SEEDS_DEG = {"gc": 573140, "mc": 829456, "mf": 261739}

# ── Shared keyword pool (engine / browser domain terms) ──
KEYWORD_POOL = [
    "asmjit", "wasm", "simd", "avx", "neon", "jit", "aot", "bytecode",
    "inlinecache", "typefeedback", "hiddenclass", "turbofan",
    "liftoff", "interpreter", "deopt", "osr", "tierup", "inlinee",
    "prototype", "closure", "scopechain", "propertycell",
    "gcroot", "safepoint", "writebarrier", "rememberedset",
    "v8", "spidermonkey", "javascriptcore", "blink",
    "compositor", "paintlayer", "scrollbar", "displayitem",
    "webrender", "audionode", "mediastream", "codec",
    "rtcdatachannel", "websocket", "xhr", "fetch",
    "renderer", "network", "storage", "dom", "css", "layout",
    "paint", "composite", "input", "events", "animation",
    "image", "video", "audio", "canvas", "webgl", "webgpu",
    "sandbox", "extension", "devtools", "inspector", "profiler",
    "memory", "performance", "accessibility", "unicode",
    "thread", "process", "ipc", "mojo", "messagepipe",
    "sync", "async", "promise", "callback", "observer",
    "validator", "serializer", "encoder", "decoder",
    "media", "worker", "serviceworker", "cache", "cookie",
    "origin", "csp", "cors", "fullscreen", "notifications",
    "geolocation", "sensors", "payment", "credential",
    "debug", "trace", "log", "monitor", "metric", "alert",
    "regression", "patch", "rollback", "merge", "build", "deploy",
    "skia", "cairo", "harfbuzz", "freetype", "subpixel",
]

COMPONENTS = [
    "Renderer", "Network", "Storage", "UI", "JavaScript", "Layout",
    "DOM", "CSS", "WebRTC", "WebGPU", "IndexedDB", "ServiceWorker",
    "Media", "Audio", "Video", "Canvas", "WebGL", "Fonts",
    "Accessibility", "Performance", "Memory", "Security", "Sandbox",
    "Extensions", "DevTools",
]

BUG_VERBS = [
    "crashes when", "fails to load", "hangs on", "renders incorrectly with",
    "throws error during", "produces wrong output for",
    "leaks memory during", "is slow when", "freezes after",
    "does not respond to", "misaligns during", "overflows with",
    "truncates data for", "ignores input from", "mishandles case of",
    "corrupts state during", "deadlocks on", "races with", "times out on",
]

BUG_NOUNS = [
    "large payloads", "concurrent requests", "nested iframes",
    "unicode characters", "service worker registration",
    "async operations", "CSS animations", "WebSocket connections",
    "cross-origin requests", "shadow DOM", "custom elements",
    "form submissions", "drag-and-drop events", "touch events",
    "high-DPI displays", "offline mode", "incognito windows",
    "large DOM trees", "recursive functions", "event propagation",
    "web workers", "indexedDB transactions", "canvas rendering",
    "media playback", "font loading", "accessibility tree",
]


def build_profiles(num_devs, keyword_pool, rng, kw_per_dev=10):
    """Assign each developer a unique set of domain keywords."""
    n_kw = len(keyword_pool)
    n_low = max(6, kw_per_dev - 3)
    n_high = kw_per_dev + 3
    personal_kw_sets = {}
    used = set()
    for dev in range(num_devs):
        n_sig = rng.randint(n_low, n_high + 1)
        for _ in range(200):
            combo = tuple(sorted(rng.sample(range(n_kw), min(n_sig, n_kw))))
            if combo not in used:
                used.add(combo)
                personal_kw_sets[dev] = set(combo)
                break
        else:
            personal_kw_sets[dev] = set(rng.sample(range(n_kw), min(n_sig, n_kw)))

    personal_profiles = np.zeros((num_devs, n_kw), dtype=np.float64)
    for dev in range(num_devs):
        for kw in personal_kw_sets[dev]:
            personal_profiles[dev, kw] = 0.5 + 0.5 * rng.random()
    personal_profiles += 0.01
    personal_profiles = personal_profiles / personal_profiles.sum(axis=1, keepdims=True)
    return personal_profiles


def _build_text(keywords, rng):
    """Build a natural-language bug report from keyword tokens."""
    component = rng.choice(COMPONENTS)
    verb = rng.choice(BUG_VERBS)
    noun = rng.choice(BUG_NOUNS)
    title = f"[{component}] {verb} {noun}"

    kw = list(keywords)
    sentences = [
        f"Observed in {component} subsystem during routine integration testing.",
        f"The issue manifests when processing {noun} under specific conditions.",
    ]
    if len(kw) >= 5:
        sentences.append(f"Diagnostic trace: {', '.join(kw[:5])}.")
    if len(kw) >= 10:
        sentences.append(f"Code paths involved: {', '.join(kw[5:10])}.")
    if len(kw) >= 15:
        sentences.append(f"Subsystems affected: {', '.join(kw[10:15])}.")
    if len(kw) >= 20:
        sentences.append(f"Modules referenced: {', '.join(kw[15:20])}.")
    if len(kw) >= 25:
        sentences.append(f"Additional dependencies: {', '.join(kw[20:25])}.")
    if len(kw) >= 30:
        sentences.append(f"Configuration entries: {', '.join(kw[25:30])}.")
    if len(kw) >= 35:
        sentences.append(f"Log indicators: {', '.join(kw[30:35])}.")
    if len(kw) >= 40:
        sentences.append(f"Stack trace symbols: {', '.join(kw[35:40])}.")
    if len(kw) >= 45:
        sentences.append(f"Further evidence: {', '.join(kw[40:45])}.")

    if len(kw) >= 3:
        emphasis = " ".join(kw[:3])
        sentences.append(
            f"Root cause analysis points to interactions between "
            f"{emphasis}. These components are critical for the observed behavior."
        )

    sentences.append(
        f"Reproduced on trunk build revision {rng.randint(100000, 999999)}. "
        f"A detailed investigation is warranted to identify the root cause."
    )
    description = " ".join(sentences)
    return title, description


def generate_bug(personal_profile, text_variation, keyword_pool, rng):
    """
    Generate a bug report by sampling keywords from the target developer's
    profile mixed with random keywords from the shared pool.

    text_variation controls the fraction of keywords drawn from random
    (non-target) sources, simulating the overlapping expertise patterns
    observed in real-world issue trackers.
    """
    kw_list = list(keyword_pool)
    n_kw = rng.randint(25, 45)
    n_signal = int(n_kw * (1 - text_variation))
    n_noise = n_kw - n_signal

    # Keywords from target developer profile
    probs = personal_profile.copy()
    threshold = probs.max() * 0.3
    probs[probs < threshold] = 0
    if probs.sum() > 0:
        probs = probs / probs.sum()
        signal_kw = list(dict.fromkeys(
            rng.choices(kw_list, weights=probs, k=n_signal * 2)))[:n_signal]
    else:
        signal_kw = []

    # Keywords from the shared pool (simulates cross-domain expertise overlap)
    noise_kw = rng.sample(kw_list, min(n_noise, len(kw_list)))

    chosen = signal_kw + noise_kw
    rng.shuffle(chosen)

    if not chosen:
        chosen = rng.sample(kw_list, min(n_kw, len(kw_list)))
    return _build_text(chosen, rng)


def generate_dataset(name, output_dir="data"):
    """Generate a synthetic dataset with the given configuration."""
    cfg = CONFIGS[name]
    num_bugs = cfg["num_bugs"]
    num_devs = cfg["num_devs"]
    text_var_lo = cfg["text_variation_lo"]
    text_var_hi = cfg["text_variation_hi"]
    var_split = cfg["variation_split"]
    label_noise = cfg["label_noise"]

    rng = random.Random(_FIXED_SEEDS[name])
    keyword_pool = list(dict.fromkeys(KEYWORD_POOL))

    personal_profiles = build_profiles(num_devs, keyword_pool, rng,
                                       kw_per_dev=cfg["kw_per_dev"])

    provider = "chromium.org" if name == "gc" else "mozilla.com"
    dev_emails = [f"developer{i:03d}@{provider}" for i in range(num_devs)]

    # Near-uniform degree distribution
    base_per_dev = num_bugs // num_devs
    remainder = num_bugs % num_devs
    bug_counts = np.full(num_devs, base_per_dev, dtype=int)
    extra_devs = rng.sample(range(num_devs), remainder)
    for d in extra_devs:
        bug_counts[d] += 1
    vrng = np.random.RandomState(_FIXED_SEEDS_DEG[name])
    variation = vrng.randint(-max(1, base_per_dev // 6),
                             max(1, base_per_dev // 6) + 1, num_devs)
    bug_counts = np.maximum(bug_counts + variation, 5)
    diff = num_bugs - bug_counts.sum()
    if diff > 0:
        for _ in range(diff):
            bug_counts[rng.randint(0, num_devs - 1)] += 1
    elif diff < 0:
        for _ in range(-diff):
            idx = rng.randint(0, num_devs - 1)
            if bug_counts[idx] > 5:
                bug_counts[idx] -= 1

    bug_to_dev = []
    for dev, count in enumerate(bug_counts):
        bug_to_dev.extend([dev] * count)
    rng.shuffle(bug_to_dev)

    # Label noise: randomly reassign a fraction of training edges
    n_random = int(len(bug_to_dev) * label_noise)
    random_indices = rng.sample(range(len(bug_to_dev)), n_random)
    for idx in random_indices:
        bug_to_dev[idx] = rng.randint(0, num_devs - 1)
    rng.shuffle(bug_to_dev)

    dev_deg = np.bincount(bug_to_dev, minlength=num_devs)
    print(f"  Degree: min={dev_deg.min()}, max={dev_deg.max()}, "
          f"mean={dev_deg.mean():.1f}, std={dev_deg.std():.1f}")

    # Split bugs into low/high text-variation tiers
    n_lo = int(num_bugs * var_split)
    is_lo_var = [True] * n_lo + [False] * (num_bugs - n_lo)
    rng.shuffle(is_lo_var)

    # Generate all records
    records = []
    for i, dev_id in enumerate(bug_to_dev):
        variation_level = text_var_lo if is_lo_var[i] else text_var_hi
        title, description = generate_bug(
            personal_profiles[dev_id], variation_level, keyword_pool, rng)
        records.append({
            "owner": dev_emails[dev_id],
            "issue_title": title,
            "description": description,
        })

    output_path = os.path.join(output_dir, f"{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[{name}] Generated {len(records)} records, "
          f"{num_devs} developers → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LANTERN datasets.")
    parser.add_argument("--name", type=str, default=None,
                        choices=["gc", "mc", "mf", None])
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    if args.name:
        generate_dataset(args.name, args.output_dir)
    else:
        for ds in ["gc", "mc", "mf"]:
            generate_dataset(ds, args.output_dir)
