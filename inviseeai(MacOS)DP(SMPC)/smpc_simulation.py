from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

from docker_smpc_manager.docker_deployer import IMAGE_NAME, build_image


class DockerSMPCError(RuntimeError):
    """Raised when a Docker-backed SMPC simulation command fails."""


@dataclass
class PartyOutcome:
    """One participant's contribution to the secure-sum protocol."""

    party_id: int
    private_value: float
    incoming_mask: float
    outgoing_mask: float
    masked_contribution: float
    seed: int


@dataclass
class SimulationBreakdown:
    """Aggregated results returned to the Flask layer for presentation."""

    parties: List[PartyOutcome]
    masked_sum: float
    recovered_sum: float
    actual_sum: float
    initial_mask: float
    final_mask: float
    explanation: List[str]
    seed: Optional[int]
    noise_range: float


def _run_party_container(
    *,
    value: float,
    incoming_mask: float,
    party_id: int,
    seed: int,
    noise_range: float,
    allow_retry: bool = True,
) -> dict:
    """Invoke a Docker container that emulates a single party in the ring."""
    cmd = [
        "docker",
        "run",
        "--rm",
        IMAGE_NAME,
        "python",
        "secret_sharing_node.py",
        "--party-id",
        str(party_id),
        "--value",
        str(value),
        "--incoming-mask",
        str(incoming_mask),
        "--seed",
        str(seed),
        "--noise-range",
        str(noise_range),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        missing_script = "python: can't open file 'secret_sharing_node.py'" in stderr
        missing_image = "unable to find image" in stderr or "pull access denied" in stderr or "manifest unknown" in stderr
        missing_binary = "no such file or directory" in stderr or "is not recognized" in stderr or "not found" in stderr

        if allow_retry and (missing_script or missing_image or missing_binary):
            build_image()
            return _run_party_container(
                value=value,
                incoming_mask=incoming_mask,
                party_id=party_id,
                seed=seed,
                noise_range=noise_range,
                allow_retry=False,
            )

        message = (
            f"Party {party_id} container failed (exit {exc.returncode}). "
            f"stdout: {exc.stdout!r} stderr: {exc.stderr!r}"
        )
        raise DockerSMPCError(message) from exc

    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DockerSMPCError(
            f"Party {party_id} returned invalid JSON: {stdout!r}"
        ) from exc
    return payload


def simulate_secure_sum(
    values: Iterable[float],
    *,
    seed: Optional[int] = None,
    rebuild_image: bool = False,
    noise_range: float = 50.0,
) -> SimulationBreakdown:
    """Simulate an additive SMPC secure-sum protocol using Docker containers."""
    numbers = [float(v) for v in values]
    if len(numbers) < 2:
        raise ValueError("Provide at least two numeric values for the simulation.")

    if noise_range <= 0:
        raise ValueError("Noise range must be a positive number.")

    if rebuild_image:
        build_image()

    rng = random.Random(seed)
    base_seed = rng.randrange(1, 10**9)
    initial_mask = rng.uniform(-noise_range, noise_range)

    incoming_mask = initial_mask
    parties: List[PartyOutcome] = []

    for idx, value in enumerate(numbers, start=1):
        party_seed = base_seed + idx
        payload = _run_party_container(
            value=value,
            incoming_mask=incoming_mask,
            party_id=idx,
            seed=party_seed,
            noise_range=noise_range,
        )

        outgoing_mask = float(payload["outgoing_mask"])
        masked_contribution = float(payload["masked_contribution"])

        parties.append(
            PartyOutcome(
                party_id=idx,
                private_value=value,
                incoming_mask=incoming_mask,
                outgoing_mask=outgoing_mask,
                masked_contribution=masked_contribution,
                seed=party_seed,
            )
        )

        incoming_mask = outgoing_mask

    masked_sum = sum(p.masked_contribution for p in parties)
    final_mask = parties[-1].outgoing_mask
    recovered_sum = masked_sum - final_mask + initial_mask
    actual_sum = sum(numbers)

    explanation = [
        "Each party receives a mask from the previous participant, adds its private value "
        "plus a fresh random mask, and forwards the new mask to the next party.",
        "The orchestrator only ever sees the masked contributions. Because every mask is "
        "added once and subtracted once, they cancel out when the contributions are summed.",
        "To close the ring, we remember the initial mask that started the protocol and "
        "remove the final mask emitted by the last party. This yields the true sum without "
        "revealing individual inputs.",
    ]

    return SimulationBreakdown(
        parties=parties,
        masked_sum=masked_sum,
        recovered_sum=recovered_sum,
        actual_sum=actual_sum,
        initial_mask=initial_mask,
        final_mask=final_mask,
        explanation=explanation,
        seed=seed,
        noise_range=noise_range,
    )
