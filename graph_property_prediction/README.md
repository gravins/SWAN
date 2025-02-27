## Graph Property Prediction Experiment
We consider the prediction of three graph properties - Diameter, Single-Source Shortest Paths (SSSP), and node Eccentricity on synthetic graphs following the setup outlined in [_Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023_](https://github.com/gravins/Anti-SymmetricDGN/tree/main/graph_prop_pred).

## How to reproduce our results
1) In the file ```run_all.sh```:
- set the variable ```save_dir_GraphProp```, i.e., root directory that stores the ```data``` folder and the results. Default is ```./gpp_exp/```.
- set the amount of gpus to run a config through the variable ```gpus```. Values can be between 0 and 1. Default is ```0.5```.
- set the number of cpus to run a config through the variable ```cpus```. Default is ```5``` cpus.
- set the ```task``` to run
- set the gpu ids. Default is ```0```.
2) Run: ``` ./run_all.sh ```
3) Automatically read results: ``` python3 read_results.py ```

