# Diseño de integración MoE inspirado en tandem.cpp

**Estado:** propuesta técnica, todavía no implementada en el motor.  
**Referencia auditada:** `ewhacc/tandem.cpp` commit `e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683`, patch base llama.cpp `f5222f25c`.  
**Hardware objetivo de validación:** Ubuntu, RTX 3050 Laptop 4 GiB (sm_86), 14 GiB RAM.

La auditoría completa y el mapeo de algoritmos están en [LEONES: tandem.cpp audit](https://github.com/robertosantosx2/LEONES/blob/research/tandem-cpp-colibri-integration/docs/subprojects/Colibri/research/tandem-cpp-audit-2026-11-10.md).

## 1. Decisión de arquitectura

No importar `tandem.patch`: es un patch de 8.732 líneas sobre una versión específica de llama.cpp, con 35 ficheros modificados. Colibrì es un runtime C/CUDA independiente.

Portar solo estos conceptos:
1. admisión de memoria antes de poblar la caché;
2. colocación atómica de los tensores que requiere la cadena CUDA;
3. split CPU/GPU exacto de misses MoE;
4. share adaptativo con medición de enlace y CPU, más compuerta A/B;
5. telemetría por capa;
6. más adelante, lookahead y doorbell.

No habilitar el descarte aproximado de expertos (`LLAMA_MOE_TAIL_SKIP`).

## 2. Orden de prioridad

### P0 — planner conjunto de memoria (primera implementación)

El planner debe conocer, por dispositivo:

```text
VRAM total
- runtime/contexto CUDA y buffers no reclamables
- pesos densos que exige la cadena
- lm_head
- estado recurrente DeltaNet
- KV mirror (si aplica)
- scratch mínimo de la cadena y prompt
- margen de seguridad
= presupuesto máximo restante para expertos
```

La cadena Qwen3.6 solo puede activarse si las proyecciones obligatorias de todas las capas y `lm_head` están en un único dispositivo compatible. No se debe admitir una colocación parcial y descubrir después que falta `dnproj` en la capa 0.

Implementación prevista:
- construir un inventario de requisitos y bytes antes de `qt_init`;
- reservar una unidad indivisible para la cadena;
- presentar al tier solo el presupuesto residual;
- si no cabe, rechazar la cadena con una razón y continuar por el camino per-matrix/híbrido existente;
- no dejar reservas huérfanas si falla una asignación.

### P1 — placement coherente

El planificador debe generar un plan completo por dispositivo antes de subir tensores. No subir cada matriz de forma independiente si la cadena depende de que todas estén presentes en el mismo dispositivo.

Para cada capa registrar:
- tipo de capa (attention/DeltaNet);
- proyecciones requeridas;
- bytes por proyección;
- dispositivo propuesto;
- resultado de admisión;
- razón del fallback.

### P2 — híbrido decode exacto

Adaptar el algoritmo de tandem.cpp:
- LRU para decidir qué expertos siguen residentes;
- contar frecuencia de los expertos miss en el microbatch;
- seleccionar una cuota de misses para upload GPU;
- calcular los misses restantes en CPU;
- combinar resultados sin omitir contribuciones.

La primera versión debe usar share fijo y no un controlador automático. Probar 0%, 10%, 20%, 30% y 50% de misses enviados a GPU. Solo se admite el camino CPU cuando el tipo/layout/activación del experto está soportado de forma exacta; si no, todos los misses se suben a GPU.

El trabajo CPU debe usar un worker persistente y buffers con ownership explícito. No se recicla ni libera un buffer mientras haya una copia o kernel en vuelo.

### P3 — telemetría y control A/B

Medir:
- expertos hit/miss y frecuencia por capa;
- bytes subidos;
- CPU time por experto;
- H2D bandwidth y tiempo de upload;
- tiempo de espera CPU visto por GPU;
- tiempo idle CPU;
- latencia de sincronización y combinación;
- VRAM/RAM/swap/page faults.

Después de comparar ventanas con híbrido ON/OFF, habilitar share adaptativo. La regla de tandem.cpp `L/(L+kR)` sirve como punto de partida, no como verdad universal. El controlador debe tener límites, pasos pequeños, histéresis y fallback al camino sin híbrido.

### P4 — prefill y predicción (posterior)

Prefill share, lookahead top-K y preread deben implementarse después del decode híbrido. Cada predicción debe ser lossless: el routing real del modelo sigue siendo autoritativo. Registrar recall@K, bytes prefetched inútiles, cancelaciones y latencia ocultada.

### P5 — doorbell GPU↔CPU (última fase)

El protocolo tipo `ggml_moe_doorbell` elimina splits del scheduler, pero requiere memoria host pinned/mapped, fences, ACK/sequence, timeouts, descriptores de upload y teardown concurrente. No introducirlo hasta que el camino híbrido con callback host esté validado y se haya demostrado que la sincronización por capa es el cuello dominante.

## 3. Restricciones del hardware objetivo

- RTX 3050 Laptop: 4 GiB VRAM, sm_86.
- RAM: 14 GiB; swap: 31 GiB.
- Modelo Qwen3.6-35B-A3B utilizado en Colibrì: aproximadamente 23,1 GB.

No copiar todo el modelo ni todos los expertos a pinned RAM. La estrategia inicial debe conservar mmap/page cache, limitar explícitamente los buffers pinned y vigilar swap. El README de tandem.cpp documenta validación en sm_89 y sm_120, no en sm_86; sus resultados no son evidencia de rendimiento para esta GPU.

## 4. Criterios de aceptación

| Área | Criterio |
|---|---|
| Presupuesto | no hay overflow ni sobreasignación; reservas de cadena preceden a la caché |
| Cadena | todas las proyecciones requeridas y lm_head en el mismo dispositivo, o fallback limpio |
| Correctitud híbrida | mismo resultado que referencia dentro de tolerancia numérica documentada; ningún experto omitido |
| Ownership | ningún use-after-free ni reutilización antes del evento de finalización |
| Control | auto no peor que el mejor share fijo dentro de la tolerancia definida |
| Rendimiento | mejora neta en decode sin degradar TTFT/pre-fill más allá del límite acordado |
| Memoria | sin crecimiento sostenido de swap ni presión que expulse el modelo de page cache |
| Recuperación | fallo de upload/CPU worker desactiva híbrido y vuelve a ruta exacta |

## 5. Matriz de pruebas

Para el MSI, ejecutar al menos tres repeticiones por configuración y alternar el orden:

| Variante | Share misses a GPU | Ruta |
|---|---:|---|
| Baseline CPU | 0% | referencia |
| Caché GPU | 100% | misses subidos |
| Híbrido fijo A | 10% | CPU/GPU concurrente |
| Híbrido fijo B | 20% | CPU/GPU concurrente |
| Híbrido fijo C | 30% | CPU/GPU concurrente |
| Híbrido fijo D | 50% | CPU/GPU concurrente |
| Híbrido auto | adaptativo | solo tras calibración |

Guardar JSONL por petición y resumen por campaña con prompt/completion tokens, TTFT, prefill tok/s, decode tok/s, wall time, hit/miss, bytes H2D, CPU time, VRAM, RSS, RAM disponible, swap y temperatura.

## 6. Próximos cambios de código previstos

1. Añadir tipos internos de plan/reserva y tests host deterministas en `c/qwen36_tier.*`.
2. Añadir una API de consulta/registro de reservas para que `qwen36.c` pueda reservar los bytes de `qwen36_cuda_chain.h` antes de inicializar el tier.
3. Hacer que el tier reciba el presupuesto residual y no vuelva a contabilizar la reserva.
4. Añadir tests de colocación completa/rollback con fake CUDA.
5. Solo después, prototipar el split híbrido decode.

Este documento no afirma que esos cambios estén implementados. Los cambios de código deben ser commits separados en esta rama, con tests host antes de solicitar pruebas CUDA en el MSI.
