# Graph Searching with Noisy Distance Predictions
This repository contains the code necessary to run the experiments in the paper "Learning-Based Algorithms for Graph Searching Problems" as submitted to the AISTATS 2024.

Three files are included:
  - ``utils.py``: contains basic objects and methods for defining graph search problems and running algorithms from the paper.
  - ``experiments.py``: uses methods from ``utils.py`` to implement the experiments outlined in the paper.
  - ``modified_astar.py``: contains a modified version of networkx's (open source) implementation of the A* search algorithm.

``utils.py`` and ``experiments.py`` require the following Python packages as dependencies: numpy, networkx, matplotlib, and scipy. ``modified_astar.py`` requires networkx.

All figures and tables reported in the paper were created using code in ``experiments.py``. Figure 2, containing empirical evaluations of the algorithms in the paper, was generated using the function random_errors_vs_graph_family. Table 2, containing a comparison of the cost incurred by Algorithm 1 compared to the theoretical upperbound, was generated using the function performance_vs_upperbounds_table. Figure 3, containing a visual comparison of our algorithms with $A^*$ was generated specifically the function compare_with_astar.

The data summarized in Figure 2 and Table 2 are contained in the folder data_for_figures.

Code for exactly recreating Figure 2 and Table 2 can be found at the bottom of ``experiments.py``. Uncommenting lines 305-314 will generate exact reproductions using the data in data_for_figures.

Code for runing new experiments and creating new visualizations in the style of Figure 2 is also included. Uncommenting lines 288-297 will give an example of calling methods from ``experiments.py``.
