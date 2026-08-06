# Engineering Lab

> *(un copiloto de ingeniería)*

---

# Objetivo

Crear un laboratorio de ingeniería reproducible sobre Google Colab que permita trabajar durante sesiones largas utilizando modelos open source ejecutándose localmente sobre GPU T4.

El laboratorio debe:

- comprender proyectos completos;
- construir conocimiento persistente;
- razonar sobre la arquitectura;
- modificar código;
- explicar absolutamente todo lo que hace;
- permitir al usuario intervenir en cualquier momento.

El usuario nunca debe sentirse "afuera" del proceso.

---

# Principios

*casi reglas inviolables.*

## 1. Todo debe ser observable

Nada ocurre "por atrás".

Si el agente decide abrir cinco archivos...

Debe verse.

```text
Investigando...

✓ renderer.ts
✓ glow.ts
✓ viewport.ts
✓ scene.ts
✓ ui.ts
```

Si decide ignorar uno.

Debe decir

```text
No abrí audio.ts

Motivo:

No tiene referencias con Renderer.
```

---

## 2. Todo debe ser reproducible

Cada acción genera un log.

```text
history/

00031/

decision.json
prompt.md
response.md
patch.diff
```

Esto permite después de una semana poder reconstruir exactamente qué pasó.

---

## 3. Nunca modificar silenciosamente. Jamás.

Siempre

```text
Propuesta

↓

Diff

↓

Explicación

↓

Aplicar
```

---

## 4. Arquitectura antes que código

El laboratorio entiende primero

```text
Proyecto

↓

Módulos

↓

Relaciones

↓

Archivos

↓

Funciones
```

No empieza leyendo líneas aleatorias.

---

# Arquitectura

```text
engineering-lab/

│
├── notebook.ipynb
│
├── config/
│
├── llm/
│
├── repository/
│
├── parser/
│
├── index/
│
├── graph/
│
├── retriever/
│
├── planner/
│
├── agent/
│
├── patcher/
│
├── history/
│
├── ui/
│
└── tests/
```

Cada carpeta hace UNA cosa.

---

## Config

Sólo guarda

```text
REPO_URL
MODEL
GPU
TEMPERATURE
API (opcional)
MAX_CONTEXT
```

Nada más.

---

## LLM

Acá irían todos los modelos.

```text
llm/

qwen.py
deepseek.py
glm.py
mistral.py
```

Todos implementan exactamente la misma interfaz.

```text
generate()
stream()
embed()
tokenize()
```

Entonces cambiar de modelo es literalmente cambiar una línea.

---

## Repository

Sólo Git.

```text
clone
pull
checkout
diff
commit
patch
```

No sabe nada de IA.

---

## Parser

Sólo entiende código.

Usaría Tree-sitter.

Produce

```text
Clase

↓

Funciones

↓

Imports

↓

Exports

↓

Comentarios

↓

Docstrings
```

Nada más.

---

## Index

Construye

```text
project_index.json
```

Con toda la información.

---

## Graph

Construye un grafo.

```text
Renderer

↓

Glow

↓

Viewport

↓

Scene
```

Y podés recorrerlo.

---

## Retriever

No usa IA.

Hace

```text
Pregunta

↓

Buscar símbolos

↓

Buscar imports

↓

Buscar embeddings

↓

Seleccionar archivos
```

---

## Planner

NO modifica código.

Sólo responde

> ¿Qué habría que hacer?

Ejemplo.

Agregar Bloom

Produce

```text
Plan

1. Modificar Glow
2. Modificar Renderer
3. Modificar Config
4. Actualizar README
```

Todavía no toca nada.

---

## Agent

Recién acá aparece el modelo.

Recibe

```text
Plan

+

Archivos

+

Contexto
```

Y propone cambios.

---

## Patcher

Sólo sabe escribir.

Genera

```text
diff
```

Nunca inventa.

---

## History

Guarda absolutamente todo.

```text
history/

00041/

question.md
thought.md
files.txt
response.md
patch.diff
metrics.json
```

---

# UI

```text
──────────────────────────────

Proyecto
Reward World

Modelo
Qwen Coder 32B

GPU
Tesla T4

Estado

✓ Indexado
✓ Grafo listo
✓ Retriever listo

──────────────────────────────

Chat

>

──────────────────────────────

Actividad

Leyendo Renderer.ts
Leyendo Glow.ts
Construyendo contexto
Generando patch

──────────────────────────────
```

Todo visible.

Las herramientas

No las ocultaría.

Las mostraría como si fueran herramientas de laboratorio.

```text
TOOLS

[ Buscar símbolo ]

[ Buscar referencias ]

[ Abrir archivo ]

[ Mostrar AST ]

[ Mostrar grafo ]

[ Ejecutar tests ]

[ Git Diff ]

[ Buscar TODO ]

[ Buscar FIXME ]
```

EL usuario mismo podría usarlas.

Agregaría una herramienta que no vi en ningún agente.

La llamaría **Explain**.

Ejemplo.

> ¿Por qué abriste Glow?

Respuesta.

Porque Renderer llama a

```text
updateGlow()
```

que está definida allí.

Además,

Glow modifica el shader principal.

No fue necesario abrir Audio porque no existe ninguna dependencia.

---

# Los modelos

Para una T4, yo apuntaría a modelos que den una muy buena relación calidad/rendimiento.

Una primera versión podría ofrecer tres perfiles:

| Modelo | Tamaño aprox. (GGUF cuantizado) | Rendimiento esperado en T4 | Uso recomendado |
|---------|-------------------------------:|---------------------------:|-----------------|
| Qwen2.5-Coder 14B Instruct | 8–10 GB | Muy bueno | Desarrollo general |
| DeepSeek-Coder V2 Lite | 9–12 GB | Muy bueno | Refactorización y generación |
| GLM-4.5 Air (si existe versión adecuada para inferencia local) o un Mistral orientado a código | 8–12 GB | Bueno | Segunda opinión / comparación |

No intentaría empezar con modelos de 70B: la complejidad aumenta mucho y el beneficio no suele compensar en una T4.

---

# Hay una idea que me gustaría que es fundacional al proyecto.

No quiero un agente que piense así:

> "Confía en mí."

Quiero uno que piense así:

> "Mirá mi razonamiento técnico."

Eso implica que no necesita revelar cadenas internas de razonamiento del modelo, sino producir un registro estructurado de decisiones generado específicamente para el usuario.

Por ejemplo.

```text
Plan de investigación

✓ Busqué la definición de Renderer.

✓ Encontré una dependencia con Glow.

✓ Verifiqué que Config controla la intensidad.

✓ No encontré referencias en Audio.

Conclusión:
Los cambios afectan únicamente tres archivos.
```

Ese tipo de explicación es suficiente para seguir el trabajo, depurar problemas y aprender la arquitectura, sin convertir el sistema en una "caja negra".

---

# Cómo dividiría el desarrollo

En lugar de intentar construir un sistema enorme desde el principio, haría versiones muy pequeñas y funcionales.

| Versión | Objetivo |
|---------|----------|
| **v0.1 – Fundación** | configuración, descarga del modelo, clonación del repositorio, chat con el modelo y visualización del estado. |
| **v0.2 – Indexador** | Tree-sitter, extracción de símbolos, índice persistente y grafo básico de dependencias. |
| **v0.3 – Herramientas** | búsqueda de símbolos, referencias, AST, exploración del repositorio y panel de actividad. |
| **v0.4 – Planificador** | el modelo propone un plan de cambios sin modificar archivos. |
| **v0.5 – Parches** | generación de git diff, revisión y aplicación controlada de cambios. |
| **v0.6 – Memoria** | historial de sesiones, reutilización del índice y conocimiento acumulado del proyecto. |

Creo que ese enfoque tiene muchas más probabilidades de producir una herramienta robusta que intentar construir un "agente autónomo" completo desde el primer día. Cada versión aporta una capacidad útil, es fácil de probar y mantiene el sistema comprensible para quien quiera estudiarlo o modificarlo.