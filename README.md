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

**Run experiments from scratch**
- Dataset generation: run the following command (one per case study)
    python data_generation/AutomAnaesthesiaDelivery.py 
    python data_generation/ExpHeatedTank.py
    python data_generation/generate_multiroom_datasets.py --nb_rooms MDOEL_DIM
    python data_generation/generate_generegul_datasets.py --nb_genes MODEL_DIM

Setting the desired number of points `nb_points` and the desired number of trajectories `nb_trajs_per_state` to simulate from each state.


- Inference:

Run the following command with the details specific of the case study considered

```
    python exec_qpm.py --model_prefix MODEL_PREFIX --model_dim MODEL_DIM --property_idx CONFIG_ID --qr_training_flag True
```

`MODEL_PREFIX` allowed are 'GRN', 'AAD' and 'EHT'. For 'AAD' and 'EHT' set the property_idx is the `CONFIG_ID` defined before.

For combining different monitors (conjunction of properties) run the following command
```
    python exec_comb_qpm.py --model_prefix MDOEL_PREFIX --model_dim MODEL_DIM --comb_idx 0 --qr_training_flag True --comb_calibr_flag True
```

The `comb_calibr_flag` is set to false is we want to train the CQR for the conjunction and set to true is we want to combine the property-specific prediction intervals. `comb_idx` enumerates the possible combinations of properties (without repetions) with the order given by flattening the non-zero elements of the upper-triangolar matrix (no diagonal included).

For sequential experiments run

    *_sequential_test.py



