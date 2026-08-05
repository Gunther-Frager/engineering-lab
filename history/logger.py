"""
history/logger.py

Implementa el principio "todo debe ser reproducible" desde v0.1,
no recien en v0.6: cada interaccion con el modelo se guarda en su
propia carpeta numerada (prompt, respuesta, config usada), para poder
reconstruir despues exactamente que paso.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


def _default_history_dir() -> Path:
    if Path("/content/drive").exists():
        return Path("/content/drive/MyDrive/engineering-lab/history")
    return Path("./history")


class HistoryLogger:

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or _default_history_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        existing = sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        n = int(existing[-1]) + 1 if existing else 1
        return f"{n:05d}"

    def log(self, prompt: str, response: str, config: dict) -> str:
        entry_id = self._next_id()
        entry_dir = self.base_dir / entry_id
        entry_dir.mkdir(parents=True)

        (entry_dir / "prompt.md").write_text(prompt)
        (entry_dir / "response.md").write_text(response)
        (entry_dir / "decision.json").write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }, indent=2))

        return entry_id
