1. Install Anaconda/Minicoda prompt
2. Go to terminal
3. Enter conda create --name cda python=3.10
4. Enter conda activate cda
5. Run the following
conda install -c conda-forge tensorflow
conda install -c conda-forge keras
conda install -c conda-forge matplotlib
conda install -c conda-forge pandas
conda install -c conda-forge tqdm
conda install pytorch=1.12.1


To train models on trace files, run in the terminal in the cda environment:
python train.py

To predict, run 
python main.py