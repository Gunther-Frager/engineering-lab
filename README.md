# Engineering Lab — v0.1 (Fundacion)

Laboratorio de ingenieria reproducible sobre Google Colab, para trabajar con
modelos open source corriendo localmente sobre GPU T4.

## Alcance de v0.1

- Configuracion centralizada (`config/`)
- Descarga y carga del modelo (`llm/`, implementacion de Qwen2.5-Coder-14B)
- Clonado del repositorio a analizar (`repository/`)
- Chat basico con el modelo, logueado en `history/`
- Panel de estado (`ui/`)

Lo que **no** esta en v0.1 (a proposito): parser, index, graph, retriever,
planner, agent, patcher. Eso es v0.2 en adelante, siguiendo el roadmap
original.

## Como correrlo

1. Subi esta carpeta a Google Drive (por ejemplo a `MyDrive/engineering-lab`)
   o subila directamente a la sesion de Colab.
2. Abri `notebook.ipynb` en Colab.
3. Runtime → Change runtime type → GPU → T4.
4. Corre las celdas en orden. La celda 0 monta Drive: es lo que hace que
   `history/`, `config.json` y el modelo descargado sobrevivan si Colab
   corta la sesion.
5. En la celda 3 poné el `REPO_URL` del repositorio que querés analizar.

## Persistencia

Si `/content/drive` esta montado, `config/`, `llm/` y `history/` guardan todo
ahi (`MyDrive/engineering-lab/...`) en vez de en el disco efimero de la
sesion. Sin esto, reiniciar el runtime borra el modelo descargado y el
historial — por eso el montaje de Drive es la celda 0, no un detalle
opcional.

## Cambiar de modelo

Todo modelo implementa `llm.base.LLMInterface`
(`generate`, `stream`, `embed`, `tokenize`). Para usar otro modelo:

1. Crear `llm/otro_modelo.py` implementando la interfaz.
2. Cambiar el import en la celda 5 del notebook.

Nada mas del laboratorio necesita saber que modelo hay detras.

## Notas de esta version

- `embed()` todavia no esta implementado en `QwenCoder` — un modelo de
  codigo no es necesariamente un buen modelo de embeddings. Se decide
  en v0.2, cuando aparece el retriever.
- `GitRepository.apply_patch_check()` ya existe (valida que un diff
  aplicaria limpio) aunque el patcher que lo va a usar recien llega en
  v0.5. Se dejo ahora porque es la pieza que hace cumplir "nunca
  modificar silenciosamente" mas adelante, sin tener que tocar
  `repository/` de nuevo.

## Roadmap

- v0.2 — Indexador: Tree-sitter, extraccion de simbolos, grafo basico.
- v0.3 — Herramientas: busqueda de simbolos, referencias, AST, panel de actividad.
- v0.4 — Planificador: propone plan de cambios sin tocar archivos.
- v0.5 — Parches: genera y aplica diffs de forma controlada.
- v0.6 — Memoria: historial de sesiones, conocimiento acumulado del proyecto.
