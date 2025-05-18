.PHONY: run clean

all: run build sim smith bimodal gshare hybrid

PYTHON = python3
PIP = pip3

run: 
	activate
	$(PYTHON) branch_sim/sim.py
	
activate: 
	requirements.txt
	python3 -m venv branch_sim_VirE
	$(PIP) install -r requirements.txt
   

build: setup.py
	python3 setup.py build bdist_wheel

sim:	branch_sim/sim.py
	echo "sim.py modified"
	touch $@

smith:	branch_sim/smith.py
	echo"smith.py modified"
	touch $@

bimodal:	branch_sim/bimodal.py
	echo "bimodal.py modified"
	touch $@

perceptron:	branch_sim/perceptron.py
	echo "perceptron.py modified"
	touch $@

gshare:	branch_sim/gshare.py
	echo "gshare.py modified"
	touch $@

hybrid: branch_sim/hybrid.py
	echo "hybrid.py modified"
	touch $@

clean:
	rm -rf build
	rm -rf dist
	rm -rf venv
	rm -rf __pycache__
