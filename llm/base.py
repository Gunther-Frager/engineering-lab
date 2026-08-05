"""
llm/base.py

Interfaz que TODO modelo del laboratorio debe implementar.
Cambiar de modelo = cambiar config.MODEL + el import correspondiente.
Ningun otro modulo del laboratorio debe saber que modelo hay detras.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List


class LLMInterface(ABC):

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Genera una respuesta completa, no incremental."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> Iterator[str]:
        """Genera la respuesta en chunks, para mostrarla en vivo en el chat."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Devuelve el embedding del texto. No se usa hasta v0.2 (retriever)."""
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Devuelve los tokens del texto. Util para no exceder MAX_CONTEXT."""
        raise NotImplementedError
