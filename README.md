# Conformal_QPM
Conformal inference for quantitative predictive monitoring of stochastic processes

## Setting up the working environment

Create a working virtual environment

1. create a virtual environment

```
pip install virtualenv
python3 -m venv qpm_env
source qpm_env/bin/activate
```
    
2. install the specified requirements
  
```
pip install -r requirements.txt
```
    

For the pcheck library, download it from: https://github.com/simonesilvetti/pcheck and install it (making sure that the `pcheck/` directory is not nested in other directories).

```
cd pcheck
python setup.py install
cd ..
```
