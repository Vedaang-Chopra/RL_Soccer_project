import os
import socket
from typing import Optional


DEFAULT_UNITY_BASE_PORT = int(os.environ.get("SOCCER_TWOS_BASE_PORT", "50039"))
_next_unity_base_port = DEFAULT_UNITY_BASE_PORT


def patch_unity_environment_close() -> bool:
    """
    ML-Agents 0.27 registers UnityEnvironment._close with atexit before
    _communicator is assigned. If environment construction fails early, atexit
    can otherwise raise AttributeError and obscure the real startup error.
    """
    try:
        from mlagents_envs.environment import UnityEnvironment
    except Exception:
        return False

    original_close = UnityEnvironment._close
    if getattr(original_close, "_soccer_twos_safe_close", False):
        return True

    def safe_close(self, timeout=None):
        if getattr(self, "_communicator", None) is not None:
            return original_close(self, timeout)

        if timeout is None:
            timeout = getattr(self, "_timeout_wait", 0)
        if hasattr(self, "_loaded"):
            self._loaded = False

        process = getattr(self, "_process", None)
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self._process = None
        return None

    safe_close._soccer_twos_safe_close = True
    safe_close._soccer_twos_original_close = original_close
    UnityEnvironment._close = safe_close
    return True


def port_is_available(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def find_free_port_block(
    start_port: Optional[int] = None,
    count: int = 1,
    max_tries: int = 1000,
    host: str = "localhost",
) -> int:
    global _next_unity_base_port
    start = int(start_port or _next_unity_base_port)
    count = max(1, int(count))

    for base_port in range(start, start + max_tries):
        if all(port_is_available(base_port + offset, host=host) for offset in range(count)):
            _next_unity_base_port = base_port + count
            return base_port
    raise OSError(
        "Could not find a free Unity base port block of size {} starting at {}".format(
            count,
            start,
        )
    )
