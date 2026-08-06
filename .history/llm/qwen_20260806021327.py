"""
llm/qwen.py

Implementacion de LLMInterface para Qwen2.5-Coder-14B-Instruct,
cuantizado en GGUF, corriendo localmente con llama-cpp-python
sobre GPU T4 (offload de capas a GPU).

Descarga el modelo la primera vez y lo cachea en disco (Drive si esta montado).
"""

from pathlib import Path
from typing import Iterator, List
import os
import shutil
import importlib
import importlib.util

# Paquetes pip que traen, cada uno, una libreria .so que llama-cpp-python
# (build cu121) necesita en tiempo de carga. Se van agregando ac medida que
# aparecen (cudart, cublas...); si en el futuro pide otra libX.so mas, se
# suma el paquete nvidia-libX-cu12 correspondiente a esta lista.
NVIDIA_LIB_PACKAGES = ["nvidia.cuda_runtime", "nvidia.cublas"]


def _ensure_cuda_runtime_available() -> None:
    """
    El wheel precompilado de llama-cpp-python (cu121) busca sus .so de CUDA
    en la misma carpeta que libllama.so (via rpath $ORIGIN), no en
    LD_LIBRARY_PATH. La solucion confiable es copiarlos ahi directo,
    ANTES de importar llama_cpp (importarlo es lo que dispara la carga y
    el crash, asi que localizamos su carpeta sin importarlo de verdad).
    """
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or not spec.submodule_search_locations:
        return
    llama_lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
    if not llama_lib_dir.exists():
        return

    for package_name in NVIDIA_LIB_PACKAGES:
        try:
            module = importlib.import_module(package_name)
            lib_dir = Path(list(module.__path__)[0]) / "lib"
        except (ImportError, IndexError):
            print(
                f"Aviso: no se encontro el paquete {package_name.replace('.', '-')}-cu12. "
                f"Instalar con: !pip install -q {package_name.replace('.', '-')}-cu12"
            )
            continue

        if not lib_dir.exists():
            continue

        for so_file in lib_dir.glob("*.so*"):
            target = llama_lib_dir / so_file.name
            if not target.exists():
                shutil.copy(so_file, target)


_ensure_cuda_runtime_available()

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from llm.base import LLMInterface

# Repo y archivo GGUF por defecto. Cambiar aca no afecta a nadie mas
# que a este archivo: esa es la idea de la interfaz comun.
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
        n_gpu_layers: int = -1,   # -1 = offload todo lo que entre en la T4
        n_ctx: int = 8192,
        verbose: bool = False,
    ):
        models_dir = _default_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        # Siempre llamamos a hf_hub_download: si el archivo ya esta completo,
        # no vuelve a bajar nada (lo detecta el propio huggingface_hub). Si
        # quedo incompleto por un corte de conexion, esto SI lo retoma o
        # rehace en vez de darlo por bueno solo porque el archivo existe.
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
        # Qwen-Coder no esta pensado como modelo de embeddings.
        # El metodo existe para cumplir la interfaz; v0.1 no lo llama.
        # En v0.2 el retriever probablemente instancie un modelo de
        # embeddings aparte (mas chico) en vez de reusar este.
        raise NotImplementedError("embed() se implementa recien en v0.2 (retriever)")

    def tokenize(self, text: str) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"))