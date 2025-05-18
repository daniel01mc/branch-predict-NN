# Branch Predictor using Neural Networks


Project directory:

branch_sim_VirE
- ------>branch_sim

              |---> __init__.py
              |--->bimodal.py
              |--->gshare.py
              |--->sim.py
              |--->smith.py
              |--->setup.py
              |--->hybrid.py
              |--->Makefile


branch_sim_VirE is a virtual envinroment (which can be modified by user)
One needs to get in that directory (cd branch_sim_VirE)
and type "make" to build project

Once project is build: run command line as 
python3 branch_sim/sim.py <predictor_model> <variable1> <variable2> traces/<tracefile

For example to run Bimodal 3: 
python3 branch_sim/sim.py bimodal 3 traces/gcc_trace.txt
################################################################################

type "make build" to build whole project
type "make clean" to remove build
##################################################################################

- The Neural networks folder contains both multi-layer neural networks and single layer perceptrons. All of those folders have their own readme files.

- There is a annotated report witht the results. 
