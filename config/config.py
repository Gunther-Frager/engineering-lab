"""
config/config.py

Configuracion central del laboratorio de ingenieria.
Esta es la UNICA fuente de verdad para parametros globales.
No contiene logica de negocio: solo valores y su carga/guardado.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


def _default_config_path() -> Path:
    # Si hay Drive montado (Colab), persistimos ahi para sobrevivir
    # a que se caiga la sesion. Si no, caemos a disco local.
    drive_dir = Path("/content/drive/MyDrive/engineering-lab")
    if Path("/content/drive").exists():
        return drive_dir / "config.json"
    return Path("./config.json")


CONFIG_PATH = _default_config_path()


@dataclass
class LabConfig:
    REPO_URL: str = ""
    MODEL: str = "qwen2.5-coder-14b-instruct"
    GPU: str = "T4"
    TEMPERATURE: float = 0.2
    API: str = ""          # opcional: fallback remoto si no hay GPU disponible
    MAX_CONTEXT: int = 8192

    def save(self, path: Path = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path = None) -> "LabConfig":
        path = path or CONFIG_PATH
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        cfg = cls()
        cfg.save(path)
        return cfg


def get_config() -> LabConfig:
    """Punto unico de acceso a la config desde el resto del laboratorio."""
    return LabConfig.load()
