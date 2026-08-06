"""
repository/git_repo.py

Todo lo que el laboratorio sabe sobre Git. No sabe nada de IA,
no interpreta codigo, no toma decisiones: solo ejecuta comandos de Git
y devuelve texto plano.
"""

import subprocess
from pathlib import Path
from typing import Optional


class GitRepository:

    def __init__(self, repo_url: str, local_path: str = "./workspace/repo"):
        self.repo_url = repo_url
        self.local_path = Path(local_path).resolve()

    def _run(self, args: list, cwd: Optional[Path] = None) -> str:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or self.local_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} fallo:\n{result.stderr}")
        return result.stdout.strip()

    def clone(self) -> str:
        if self.local_path.exists():
            # Ignoramos archivos ocultos (.gitkeep, .gitignore, etc.) al decidir
            # si la carpeta esta "vacia": para Git cuentan como contenido y
            # rechaza clonar ahi, pero logicamente no hay nada real todavia.
            visible_contents = [
                p for p in self.local_path.iterdir()
                if not p.name.startswith(".")
            ]
            if visible_contents:
                return f"Ya existe {self.local_path}, uso pull() en vez de clone()."
            for hidden in self.local_path.iterdir():
                hidden.unlink()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        return self._run(
            ["clone", self.repo_url, self.local_path.name],
            cwd=self.local_path.parent,
        )

    def pull(self) -> str:
        return self._run(["pull"])

    def checkout(self, ref: str) -> str:
        return self._run(["checkout", ref])

    def diff(self, ref: Optional[str] = None) -> str:
        args = ["diff"] + ([ref] if ref else [])
        return self._run(args)

    def commit(self, message: str) -> str:
        self._run(["add", "-A"])
        return self._run(["commit", "-m", message])

    def apply_patch_check(self, patch_text: str) -> bool:
        """
        Valida que un patch aplicaria limpio, SIN aplicarlo.
        El patcher real (v0.5) usa esto antes de mostrarle el diff al usuario.
        """
        patch_file = self.local_path / ".tmp_patch.diff"
        patch_file.write_text(patch_text)
        try:
            self._run(["apply", "--check", patch_file.name])
            return True
        except RuntimeError:
            return False
        finally:
            patch_file.unlink(missing_ok=True)

    def status(self) -> str:
        return self._run(["status", "--short"])
