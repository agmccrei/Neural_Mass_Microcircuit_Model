# Mean Field Microcircuit Model
===================================================================================================================================================================
Author: Alexandre Guet-McCreight

This is the readme for usage of the mean field model described within circuit0_definitions.py.

To run model optimization:
python circuit1_optimization.py

To analyze best fit models from optimization:
python circuit2_analysis.py

To plot optimized model parameters:
python circuit3_plot_optimized_params.py

To run sensitivity analysis on the optimized models, removing each population and re-simulating in a step-wise fashion:
python circuit4_removepops.py

NOTE: ipynb versions exist for circuit2_analysis.py and circuit4_removepops.py.