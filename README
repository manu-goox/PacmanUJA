En esta página se solicita implementar varios algoritmos de búsqueda para un proyecto de Pacman en Python. Aquí están las tareas principales:

1. **Configuración inicial**:
   - Descargar y preparar el código fuente proporcionado.
   - Probar el entorno con comandos básicos (`python pacman.py`) para familiarizarse con cómo funciona Pacman.

2. **Tareas específicas**:
   - **Q1: Depth First Search (DFS)**: Implementar el algoritmo de búsqueda en profundidad en el archivo `search.py`. Verificar su funcionamiento probándolo en diferentes laberintos como `tinyMaze` y `mediumMaze`.
   - **Q2: Breadth First Search (BFS)**: Implementar la búsqueda en anchura. Nuevamente, probar el código para asegurar que encuentra las soluciones esperadas.
   - **Q3: Uniform Cost Search (UCS)**: Modificar el algoritmo para encontrar caminos que optimicen un costo dado, como evitar áreas peligrosas o priorizar áreas con comida.
   - **Q4: A* Search**: Implementar el algoritmo A* usando heurísticas como la distancia Manhattan para guiar la búsqueda.

3. **Herramientas útiles**:
   - Utilizar estructuras de datos específicas como `Stack`, `Queue` y `PriorityQueue` de `util.py`.
   - Asegurarse de escribir versiones de búsqueda basadas en grafo para evitar expandir nodos visitados previamente.

4. **Evaluación y pruebas**:
   - Ejecutar pruebas automáticas con `python autograder.py` para evaluar las soluciones.
   - Comprobar resultados visuales en los laberintos para validar el comportamiento del agente.

5. **Extras**:
   - Familiarizarse con las anotaciones de tipos en Python para entender el código.
   - Explorar configuraciones opcionales y parámetros del script (`python pacman.py -h`).

El objetivo principal es implementar y probar algoritmos de búsqueda que permitan a Pacman navegar por los laberintos de manera óptima bajo diferentes criterios.


---

# Tareas concretas del Pacman en la tarea de Berkeley


### **Q1: Depth First Search (DFS)**
- Implementar el algoritmo de **búsqueda en profundidad** (DFS) en el archivo `search.py`.
- Este algoritmo debe expandir los nodos explorando en profundidad antes de retroceder.
- Probarlo en laberintos como `tinyMaze` para garantizar que encuentra un camino válido para que Pacman alcance la meta.
- Utilizar una estructura tipo **pila (stack)** para manejar los nodos pendientes.

---

### **Q2: Breadth First Search (BFS)**
- Implementar **búsqueda en anchura** (BFS) en `search.py`.
- Este algoritmo debe expandir los nodos más cercanos primero, asegurando encontrar el camino más corto (en términos de número de pasos).
- Probarlo en laberintos como `mediumMaze` y `bigMaze`.
- Utilizar una estructura tipo **cola (queue)** para manejar los nodos pendientes.

---

### **Q3: Uniform Cost Search (UCS)**
- Implementar el algoritmo de **búsqueda de costo uniforme** (UCS).
- Este algoritmo debe priorizar los caminos según su costo acumulado (definido como el costo de transición entre estados).
- Utilizar una **cola de prioridad (priority queue)** para manejar los nodos pendientes.
- Probarlo en laberintos que incluyan un costo asociado a las transiciones.

---

### **Q4: A* Search**
- Implementar el algoritmo **A* Search**, que combina UCS con una función heurística para guiar la búsqueda.
- Utilizar la distancia Manhattan como heurística inicial.
- Probarlo en los mismos laberintos que UCS y comparar el rendimiento.
- Especificar la heurística en el archivo `searchAgents.py`.

---

### **Q5: Corners Problem**
- Diseñar una solución para que Pacman recoja comida en las esquinas de un laberinto.
- Este problema requiere encontrar un camino que visite las cuatro esquinas del mapa.
- Implementar una nueva función de búsqueda y probarla con `python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem`.

---

### **Q6: Corners Heuristic**
- Diseñar una **heurística admisible y consistente** para resolver el problema de las esquinas.
- La heurística debe estimar el costo mínimo restante para visitar todas las esquinas no visitadas.
- Probar la solución con los comandos proporcionados para garantizar eficiencia.

---

### **Q7: Eating All the Dots (Food Search Problem)**
- Resolver el problema en el que Pacman debe comer todos los puntos de comida en el mapa.
- Implementar una búsqueda óptima para recolectar toda la comida usando BFS o A*.
- Probar el agente con `python pacman.py -l trickySearch`.

---

### **Q8: Suboptimal Search**
- Diseñar un agente más rápido y posiblemente subóptimo para recolectar comida.
- Este agente no necesita garantizar la solución óptima, pero debe ser eficiente en tiempo.
- Usar heurísticas o estrategias personalizadas para encontrar un balance entre rapidez y calidad del camino.

---