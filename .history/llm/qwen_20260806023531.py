"""
llm/qwen.py

Implementacion de LLMInterface para Qwen2.5-Coder-14B-Instruct,
cuantizado en GGUF, corriendo localmente con llama-cpp-python
sobre GPU T4 (offload de capas a GPU).

Descarga el modelo la primera vez y lo cachea en disco (Drive si esta montado).
"""

from pathlib import Path
from typing import Iterator, List
import shutil
import site
import importlib.util


def _ensure_cuda_libs_available() -> None:
    """
    El wheel precompilado de llama-cpp-python (cu121) busca varias .so de
    CUDA (cudart, cublas, y las que hagan falta) en la misma carpeta que
    libllama.so, via rpath $ORIGIN. En vez de agregarlas de a una a medida
    que aparece el proximo error, copiamos TODAS las que ya estan instaladas
    en el sistema (Colab las trae como dependencia de torch, preinstalado).
    """
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or not spec.submodule_search_locations:
        return
    llama_lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
    if not llama_lib_dir.exists():
        return

    copied_any = False
    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for so_file in nvidia_dir.glob("*/lib/*.so*"):
            target = llama_lib_dir / so_file.name
            if not target.exists():
                shutil.copy(so_file, target)
            copied_any = True

    if not copied_any:
        print(
            "Aviso: no se encontraron librerias nvidia instaladas. "
            "Instalar torch (trae el stack CUDA completo) con: !pip install -q torch"
        )


_ensure_cuda_libs_available()

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from llm.base import LLMInterface

DEFAULT_REPO_ID = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"


def _default_models_dir() -> Path:
    if Path("/content/drive").exists():
        return Path("/content/drive/MyDrive/engineering-lab/models")
    return Path("./models")


class QwenCoder(LLMInterface):

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        verbose: bool = False,
    ):
        models_dir = _default_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        print(f"Verificando/descargando {filename} desde {repo_id}...")
        model_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_dir),
        ))

        print(f"Cargando modelo en GPU (n_gpu_layers={n_gpu_layers})...")
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self.n_ctx = n_ctx

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return out["choices"][0]["message"]["content"]

    def stream(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> Iterator[str]:
        chunks = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("embed() se implementa recien en v0.2 (retriever)")

    def tokenize(self, text: str) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"))