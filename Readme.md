# Análisis del algoritmo de gradiente descendente y estudio empírico comparativo con técnicas metaheurísticas

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Notebook-Jupyter-orange)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- Desarrollado en su totalidad por Eduardo Morales Muñoz.
- Dirigido por Pablo Mesejo Santiago y Javier Meri de la Maza.
- Proyecto de fin de carrera del **Doble Grado en Ingeniería Informática y Matemáticas** de la **Universidad de Granada**.  
- Calificación obtenida: **10.0  (Matrícula de Honor)**

Este repositorio contiene el material completo de una tesis de investigación que estudia en profundidad el algoritmo de Gradiente Descendente (GD) y lo compara empíricamente con diversas técnicas metaheurísticas (MH) para el entrenamiento de redes neuronales profundas.

## Resumen

El trabajo se divide en dos partes principales:

-  **Parte matemática**: Estudio teórico de las condiciones de convergencia del Gradiente Descendente en sus versiones determinista y estocástica, con énfasis en el uso de teoría de martingalas como herramienta analítica en el caso estocástico.
- **Parte computacional**: Análisis experimental extensivo comparando optimizadores basados en GD y MH (incluyendo el algoritmo SHADE-ILS, con rendimiento de estado del arte), sobre tareas de clasificación y regresión, usando tanto MLPs como ConvNets, con datasets y modelos de diversa complejidad. Además se proponen dos técnicas meméticas originales que combinan el GD con MH.

Este proyecto ofrece una comprensión más precisa de las limitaciones y fortalezas de las MHs frente al GD, proporcionando un marco útil para futuras investigaciones y desarrollos de nuevos algoritmos híbridos o adaptativos.

## Estructura del repositorio

```text
├── memoria.pdf                 # Documento completo de la tesis
├── presentacion.pdf           # Diapositivas usadas en la defensa
├── codigo/                    # Código fuente en Python
│   ├── comparative.ipynb      # Análisis final de resultados
│   ├── dataset_X.ipynb        # Un notebook por dataset
│   ├── utilsTFG.py            # Funciones reutilizables para los notebooks
│   ├── *.csv, *.txt           # Datos y resultados generados
├── latex/                     # Archivos LaTeX de la memoria (compilables)
```

## Tecnologías utilizadas

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [SciPy](https://scipy.org/)
- [PyADE](https://github.com/jmmvruano/PyADE)
- `pickle` para guardado/carga de datos experimentales

---

## Cómo ejecutar los experimentos

El código está estructurado por dataset, y cada notebook puede ejecutarse de forma independiente.

1. Abre el notebook correspondiente (`dataset_X.ipynb`).
2. Ejecuta las celdas de carga y tratamiento de datos.
3. Ejecuta el bloque de código correspondiente al experimento deseado.
4. Los resultados se almacenan automáticamente como `.csv` o `.txt`.

>  La reproducibilidad está garantizada mediante fijación explícita de semillas en todos los experimentos.

>  Nota: El código no está extensamente comentado, pero su estructura es modular y fácilmente interpretable.

---

## Resultados y análisis

El estudio empírico se estructura en torno a cuatro ejes principales:

###  1. Influencia del tipo de tarea
- Se analiza el impacto de las tareas de **clasificación vs. regresión** en el rendimiento de MLPs entrenados con MH.
- El rendimiento de los modelos entrenados con GD no se ve afectado significativamente por el tipo de tarea.
- En cambio, los modelos entrenados con MH sí muestran diferencias: mediante un test de Wilcoxon (p-valor = 0.023), se concluye que **hay diferencias estadísticamente significativas en el rendimiento de las MH según la naturaleza de la tarea**, aunque esta conclusión es sensible al conjunto de optimizadores seleccionados.

###  2. Factores que afectan al rendimiento
- Se emplea un análisis de dependencias parciales sobre tareas de clasificación.
- El factor más influyente para el **rendimiento de las MH** es el **tamaño del conjunto de datos**.
- En contraste, para el entrenamiento con GD, el rendimiento depende más de la **complejidad del conjunto de datos** (número de clases, desbalanceo, estructura de la información…).

###  3. Tiempo de ejecución
- Las MH requieren **menos tiempo por época**, pero un **número de épocas mucho mayor** (hasta 100-200x más) para alcanzar resultados comparables con GD.
- Esto las convierte en una **opción globalmente más lenta**.
- El número de instancias del conjunto de datos influye más en el tiempo que la cantidad de parámetros del modelo.

###  4. Comparativa de rendimiento
- El nuevo enfoque **SHADE-GD** supera a SHADE en **17 de 25 tareas**, mostrando mayor consistencia y generalización.
- **SHADE-ILS-GD** mejora en algunas tareas concretas, pero no ofrece un rendimiento robusto en redes profundas.
- En general, **SHADE-GD** es la propuesta más prometedora entre las metaheurísticas evaluadas.
**Aportación original**: Se observa que el **rendimiento de las MH depende más del tamaño del dataset que de su complejidad** o del número de parámetros del modelo. Esta dependencia no ha sido ampliamente documentada en trabajos anteriores y representa una contribución novedosa.
---

## Conclusiones

Los resultados respaldan la **superioridad de los optimizadores basados en Gradiente Descendente** frente a las metaheurísticas en términos de eficiencia, estabilidad y rendimiento general en el entrenamiento de modelos de deep learning.

No obstante, este trabajo ofrece una visión crítica y detallada sobre las **limitaciones y el potencial real de las técnicas MH**, identificando escenarios donde podrían resultar competitivas y estableciendo un marco útil para futuras investigaciones y desarrollos híbridos.


## Autor

Realizado por [Eduardo Morales Muñoz](https://www.linkedin.com/in/eduardo-morales-264101346/)  
Dirigido por Pablo Mesejo Santiago y Javier Meri de la Maza
Graduado en Matemáticas
Graduado en Informática
Universidad de Granada · 2025

---

## Licencia

Este proyecto se publica bajo la licencia [MIT](LICENSE).

---

##  Portfolio

Este repositorio forma parte de mi portfolio profesional, como muestra de mi capacidad para:

- Realizar investigación aplicada combinando matemáticas y ciencia de datos.
- Desarrollar y evaluar algoritmos avanzados de optimización y aprendizaje profundo.
- Ejecutar experimentos reproducibles y realizar análisis comparativos rigurosos.

¡Gracias por tu interés!

