"""
ui/status.py

Panel de estado textual del laboratorio, pensado para imprimirse
en una celda de Colab. Nada oculto: si algo no esta listo, lo dice
en vez de simular que si.
"""

from config.config import LabConfig


def render_status(config: LabConfig, repo_ready: bool, model_ready: bool) -> str:
    lines = [
        "-" * 42,
        f"Proyecto   {config.REPO_URL or '(sin definir)'}",
        f"Modelo     {config.MODEL}",
        f"GPU        {config.GPU}",
        "-" * 42,
        "Estado",
        f"{'[x]' if repo_ready else '[ ]'} Repositorio clonado",
        f"{'[x]' if model_ready else '[ ]'} Modelo cargado",
        "[ ] Indexado        (llega en v0.2)",
        "[ ] Grafo listo     (llega en v0.2)",
        "[ ] Retriever listo (llega en v0.2)",
        "-" * 42,
    ]
    text = "\n".join(lines)
    print(text)
    return text
