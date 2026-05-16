"""GPU memory guard — VRAM info for system status display.

Load decisions use analytical tracking in ModelManager, not nvidia-smi.
This module only provides nvidia-smi data for health/settings display.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

TOTAL_VRAM_GB = 10.0  # RTX 3080


def get_free_vram_gb() -> float:
    """Query nvidia-smi for free GPU memory in GB. For display only."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip().split("\n")[0])
            return free_mb / 1024.0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.warning("nvidia-smi query failed: %s", e)
    return TOTAL_VRAM_GB


def get_gpu_info() -> dict:
    """Return GPU info dict for system status display."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            total_mb = int(parts[1])
            used_mb = int(parts[2])
            free_mb = int(parts[3])
            return {
                "name": parts[0],
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "vram_total_gb": round(total_mb / 1024, 2),
                "vram_used_gb": round(used_mb / 1024, 2),
                "vram_free_gb": round(free_mb / 1024, 2),
                "temperature_c": int(parts[4]),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return {
        "name": "Unknown", "total_mb": 0, "used_mb": 0, "free_mb": 0,
        "vram_total_gb": 0.0, "vram_used_gb": 0.0, "vram_free_gb": 0.0,
        "temperature_c": 0,
    }
