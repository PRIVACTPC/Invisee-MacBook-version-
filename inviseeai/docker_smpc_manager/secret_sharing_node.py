#!/usr/bin/env python
"""
Executed inside a lightweight Docker container to emulate a single SMPC party.

Given the incoming mask (from the previous party in the ring) and the party's
private value, it samples a new mask, produces the masked contribution that an
aggregator would observe, and forwards the freshly sampled mask to the next
party.
"""

import argparse
import json
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate an SMPC party step.")
    parser.add_argument("--party-id", type=int, required=True)
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--incoming-mask", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise-range", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    outgoing_mask = rng.uniform(-args.noise_range, args.noise_range)
    masked_contribution = args.value + outgoing_mask - args.incoming_mask

    payload = {
        "party_id": args.party_id,
        "outgoing_mask": outgoing_mask,
        "masked_contribution": masked_contribution,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
