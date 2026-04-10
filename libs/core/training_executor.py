"""Training executor — runs training via subprocess from a TrainingPlan."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from libs.core.contracts import TrainingPlan, TrainingResult


def execute_training(
    plan: TrainingPlan,
    timeout: int = 3600,
) -> TrainingResult:
    """Run the training command from a plan and return the result.

    Does not validate readiness — caller should run validate_training first.
    """
    start = time.monotonic()

    # Ensure output dir exists
    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)

    # Parse command into args
    # The command may contain backslash-newlines from formatting
    cmd = plan.command.replace("\\\n", " ")
    args = shlex.split(cmd)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        duration = time.monotonic() - start
        return TrainingResult(
            snapshot_id=plan.snapshot_id,
            plan_method=plan.method,
            exit_code=-1,
            success=False,
            output_dir=plan.output_dir,
            duration_sec=round(duration, 2),
            error="Training command not found. Is TRL installed? "
            "Install with: pip install cassette[training]",
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        return TrainingResult(
            snapshot_id=plan.snapshot_id,
            plan_method=plan.method,
            exit_code=-2,
            success=False,
            output_dir=plan.output_dir,
            duration_sec=round(duration, 2),
            error=f"Training timed out after {timeout}s",
        )

    duration = time.monotonic() - start
    stdout_lines = proc.stdout.strip().split("\n") if proc.stdout else []
    tail = "\n".join(stdout_lines[-20:])

    if proc.returncode != 0:
        error_msg = proc.stderr.strip()[-500:] if proc.stderr else "Unknown error"
        return TrainingResult(
            snapshot_id=plan.snapshot_id,
            plan_method=plan.method,
            exit_code=proc.returncode,
            success=False,
            output_dir=plan.output_dir,
            duration_sec=round(duration, 2),
            stdout_tail=tail,
            error=error_msg,
        )

    return TrainingResult(
        snapshot_id=plan.snapshot_id,
        plan_method=plan.method,
        exit_code=0,
        success=True,
        output_dir=plan.output_dir,
        duration_sec=round(duration, 2),
        stdout_tail=tail,
    )
