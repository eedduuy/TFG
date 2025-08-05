**English version below**

# ES: Análisis del algoritmo de gradiente descendente y estudio empírico comparativo con técnicas metaheurísticas

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



## Cómo ejecutar los experimentos

El código está estructurado por dataset, y cada notebook puede ejecutarse de forma independiente.

1. Abre el notebook correspondiente (`dataset_X.ipynb`).
2. Ejecuta las celdas de carga y tratamiento de datos.
3. Ejecuta el bloque de código correspondiente al experimento deseado.
4. Los resultados se almacenan automáticamente como `.csv` o `.txt`.

>  La reproducibilidad está garantizada mediante fijación explícita de semillas en todos los experimentos.

>  Nota: El código no está extensamente comentado, pero su estructura es modular y fácilmente interpretable.



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


## Conclusiones

Los resultados respaldan la **superioridad de los optimizadores basados en Gradiente Descendente** frente a las metaheurísticas en términos de eficiencia, estabilidad y rendimiento general en el entrenamiento de modelos de deep learning.

No obstante, este trabajo ofrece una visión crítica y detallada sobre las **limitaciones y el potencial real de las técnicas MH**, identificando escenarios donde podrían resultar competitivas y estableciendo un marco útil para futuras investigaciones y desarrollos híbridos.


## Autor

Realizado por [Eduardo Morales Muñoz](https://www.linkedin.com/in/eduardo-morales-264101346/)  

Dirigido por Pablo Mesejo Santiago y Javier Meri de la Maza

Graduado en Matemáticas

Graduado en Informática

Universidad de Granada · 2025



## Licencia

Este proyecto se publica bajo la licencia [MIT](LICENSE).


##  Portfolio

Este repositorio forma parte de mi portfolio profesional, como muestra de mi capacidad para:

- Realizar investigación aplicada combinando matemáticas y ciencia de datos.
- Desarrollar y evaluar algoritmos avanzados de optimización y aprendizaje profundo.
- Ejecutar experimentos reproducibles y realizar análisis comparativos rigurosos.

¡Gracias por tu interés!

---------




# EN: Analysis of the gradient descent algorithm and comparative empirical study with metaheuristic techniques

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Notebook-Jupyter-orange)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- Developed entirely by Eduardo Morales Muñoz.
- Directed by Pablo Mesejo Santiago and Javier Meri de la Maza.
- Final project for the **Double Degree in Computer Engineering and Mathematics** at the **University of Granada**.
- Grade obtained: **10.0  (Honors)**

This repository contains the complete material of a research thesis that studies in depth the Gradient Descent (GD) algorithm and compares it empirically with various metaheuristic (MH) techniques for training deep neural networks.

## Summary

The work is divided into two main parts:

-  **Mathematical part**: Theoretical study of the convergence conditions of Gradient Descent in its deterministic and stochastic versions, with an emphasis on the use of martingale theory as an analytical tool in the stochastic case.
- **Computational part**: Extensive experimental analysis comparing GD- and MH-based optimizers (including the state-of-the-art SHADE-ILS algorithm) on classification and regression tasks, using both MLPs and ConvNets, with datasets and models of varying complexity. In addition, two original memetic techniques combining GD with MH are proposed.

This project offers a more accurate understanding of the limitations and strengths of MHs compared to GD, providing a useful framework for future research and development of new hybrid or adaptive algorithms.

## Repository structure

```text
├── memoria.pdf                # Complete thesis document
├── presentacion.pdf           # Slides used in the defense
├── codigo/                      # Source code in Python
│   ├── comparative.ipynb      # Final analysis of results
│   ├── dataset_X.ipynb        # One notebook per dataset
│   ├── utilsTFG.py            # Reusable functions for notebooks
│   ├── *.csv, *.txt           # Generated data and results
├── latex/                     # LaTeX files of the thesis (compilable)
```

## Technologies used

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [SciPy](https://scipy.org/)
- [PyADE](https://github.com/jmmvruano/PyADE)
- `pickle` for saving/loading experimental data



## How to run the experiments

The code is structured by dataset, and each notebook can be run independently.

1. Open the corresponding notebook (`dataset_X.ipynb`).
2. Run the data loading and processing cells.
3. Run the code block corresponding to the desired experiment.
4. The results are automatically stored as `.csv` or `.txt`.

> Reproducibility is guaranteed by explicitly setting seeds in all experiments.

> Note: The code is not extensively commented, but its structure is modular and easily interpretable.



## Results and analysis

The empirical study is structured around four main axes:

###  1. Influence of task type
- The impact of **classification vs. regression** tasks on the performance of MLPs trained with MH is analyzed.
- The performance of models trained with GD is not significantly affected by task type.
- In contrast, models trained with MH do show differences: using a Wilcoxon test (p-value = 0.023), it is concluded that **there are statistically significant differences in the performance of MHs depending on the nature of the task**, although this conclusion is sensitive to the set of optimizers selected.

###  2. Factors affecting performance
- A partial dependency analysis is used on classification tasks.
- The most influential factor for **MH performance** is the **size of the dataset**.
- In contrast, for GD training, performance depends more on the **complexity of the dataset** (number of classes, imbalance, information structure, etc.).

###  3. Execution time
- MH requires **less time per epoch**, but a **much higher number of epochs** (up to 100-200x more) to achieve results comparable to GD.
- This makes it a **slower option overall**.
- The number of instances in the dataset has a greater influence on time than the number of model parameters.

###  4. Performance comparison
- The new **SHADE-GD** approach outperforms SHADE in **17 out of 25 tasks**, showing greater consistency and generalization.
- **SHADE-ILS-GD** improves on some specific tasks, but does not offer robust performance on deep networks.
- Overall, **SHADE-GD** is the most promising proposal among the metaheuristics evaluated.
**Original contribution**: It is observed that the **performance of MHs depends more on the size of the dataset than on its complexity** or the number of model parameters. This dependence has not been widely documented in previous work and represents a novel contribution.


## Conclusions

The results support the **superiority of gradient descent-based optimizers** over metaheuristics in terms of efficiency, stability, and overall performance in deep learning model training.

However, this work offers a critical and detailed view of the **limitations and real potential of MH techniques**, identifying scenarios where they could be competitive and establishing a useful framework for future research and hybrid developments.


## Author

Produced by [Eduardo Morales Muñoz](https://www.linkedin.com/in/eduardo-morales-264101346/)  

Directed by Pablo Mesejo Santiago and Javier Meri de la Maza

Graduate in Mathematics

Graduate in Computer Science

University of Granada · 2025



## License

This project is published under the [MIT](LICENSE) license.


##  Portfolio

This repository is part of my professional portfolio, demonstrating my ability to:

- Conduct applied research combining mathematics and data science.
- Develop and evaluate advanced optimization and deep learning algorithms.
- Perform reproducible experiments and conduct rigorous comparative analyses.

Thank you for your interest!

