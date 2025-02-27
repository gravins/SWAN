## Graph Transfer Experiment
This task consists on transfering a label from a source node to a target node on different graph topologies.


## How to reproduce our results
1) Uncompress ```data.zip```
2) In the file ```run_all.py``` set:
- set the variable ```root```, i.e., root directory that stores the ```data``` and ```results``` folders
- set the available ```gpus``` ids used for the experiments
- set the list of ```models``` to evaluate. The model names are: _gin_, _gcn_, _gat_, _sage_, _gps_, _adgn_, _swan_
3) Run: ``` nohup python3 -u run_all.py ```
4) Make plot: ``` python3 plot_graph_transfer_task.py ```

