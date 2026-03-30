def linear_ramp(step: int, total_steps: int, start: float, end: float) -> float:
    alpha = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return start + (end - start) * alpha
