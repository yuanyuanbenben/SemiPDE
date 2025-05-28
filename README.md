# Materials for "Deep Semiparametric Partial Differential Equation Models"

#### ***Note***: Due to size limit of uploaded files, we hereby provide the codes used in the manuscript, while all the output files, plots and dataset are put on the GitHub public repository [https://github.com/yuanyuanbenben/SemiPDE](https://github.com/yuanyuanbenben/SemiPDE). The real datasets are publicly available with links provided in the manuscript. For your ease, we also present how to preprocess them on the above GitHub public repository.

## Overview

- ### Directory ***simulation*** contains all codes, outputs and plots related to the simulations in *Section 5: Simulation*.
    - The folders ***outputs*** and ***outputs_paramodel*** store simulation results, including estimated parameters, values of the loss function, and other related outputs. The ***outputs*** folder contains results for the SemiPDE model, while ***outputs_paramodel*** contains results for parametric PDE models.
    - The folders ***checkpoint*** and ***checkpoint_paramodel*** include the checkpoints of neural network models used in the simulations. The ***checkpoint*** folder stores models estimated using the SemiPDE approach, and ***checkpoint_paramodel*** stores those for parametric PDE models.
    - The ***pic*** folder includes plots presented in the main text and supplementary materials.
    - The files ***xxx_main.py*** are the main Python scripts for Cases 1–4. Specifically,  ***RD*** represents case 1, ***GFN*** case 2, ***NP*** case 3 and ***NS*** case 4. ***RCD_compa*** represent the model used in the last simyulation. 
    - The files ***xxx_dataset.py*** generate the datasets used in the simulations for Cases 1–4.
    - The files ***xxx_model.py*** define classes for numerically solving known mechanisms, approximating unknown mechanisms using neural networks, and integrating them into the SemiPDE model.
    - The files ***xxx_optimal.py*** implement the optimization procedures for SemiPDE estimation.
    - The files ***xxx_nonpara.py*** contain functions used for implementing Baseline 2.
    - The files ***xxx_PINN_main.py*** implement Baseline 3.
    - The files ***xxx_variance.py*** are the main scripts for parameter inference in the SemiPDE framework.
    - The files ***PINN_xxx.py*** support the implementation of Baseline 3.
    - The files ***xxx.ipynb*** present selected results for estimation and inference.
    - The files ***plot.py*** and ***plot2.py*** are used to generate plots included in the paper.
    - The ***xxx.sh*** files are shell scripts used to execute the Python programs.

- ### Directory ***realdata*** contains all dataset, codes, outputs and plots related to the two real-data experiments described in *Section 6: Real Data Applications*. 
    - #### The ***realdata1*** subdirectory includes materials for *Section 6.1: The Behaviour of In Vitro Cell Culture Assays*. 
        - The ***data*** folder contains the preprocessed dataset.
        - The ***outputs*** folder stores results, including estimated parameters, values of the loss function, and other related outputs.
        - The ***checkpoint*** folder includes the checkpoints of neural network models used in estimation.
        - The ***pic*** folder includes plots presented in the main text and supplementary materials.
        - The files ***xxx.xlsx*** are raw data files from the real dataset.
        - The file ***main.py*** is the main Python script.
        - The file ***realdata_1_dataset.py*** loads the data into dataloaders used for optimization.
        - The file ***realdata_1_model.py*** defines classes for numerically solving known mechanisms, approximating unknown mechanisms using neural networks, and integrating them into the SemiPDE model.
        - The file ***realdata_1_optimal.py*** implements the optimization procedures for SemiPDE estimation.
        - The file ***realdata_1_paramodel.py*** implements baseline models.
        - The file ***plot.py*** generates the plots used in the paper.
    - #### The ***realdata2*** subdirectory includes materials for *Section 6.2: Wave Propagation Through Vegetation*. 
        - The ***data*** folder contains the preprocessed dataset.
        - The ***outputs*** folder stores results, including estimated parameters, values of the loss function, and other related outputs.
        - The ***checkpoint*** folder includes the checkpoints of neural network models used in estimation.
        - The ***pic*** folder includes plots presented in the main text and supplementary materials.
        - The file ***main.py*** is the main Python script.
        - The file ***realdata_2_dataset.py*** loads the data into dataloaders used for optimization.
        - The file ***realdata_2_model.py*** defines classes for numerically solving known mechanisms, approximating unknown mechanisms using neural networks, and integrating them into the SemiPDE model.
        - The file ***realdata_2_optimal.py*** implements the optimization procedures for SemiPDE estimation.
        - The files ***plot.py*** and ***plot2.py*** generate the plots used in the paper.


## Workflows
#### Note: Requires Python 3 and the following libraries: ***numpy***, ***torch***, ***pandas***, ***scipy.io***, ***py-pde***, ***shelve***, ***os***, ***siren_pytorch***, ***collections***, ***torchdiffeq***, ***ignite.utils***, ***GPUtil***, ***argparse*** and ***copy***. 

- ### Simulation for estimation performance of the proposed method compared to three baselines. 
    - For estimation of case 1-4, users can directly run the shell script as `./xxx_test.sh`. Some parameters may need to be adjusted for your specific setup.  
        - `-c` specifies the GPU IDs to use.
        - `-seed` sets the starting random seed for repeated runs.
        - `-mod` parameteric or semiparametric model.
    - For baseline 1, set the variable `-mod` in ***xxx_test.sh*** as `no_func`.
    - For baseline 2, set the file `xxx_nonpara.py` instead of `xxx_main.py`.
    - For baseline 3, run the shell script ***xxx_pinn.sh***.

- ### Simulation for inference of parameters in the SemiPDE model. 
    - Users can directly run `./xxx_para_variance.sh`. 

- ### Simulation for identifiability and robustness of the SemiPDE model.
    - his simulation is similar to the first one, except the true function is set to `0`.

- ### Real-data experiment 1: The Behaviour of In Vitro Cell Culture Assays
    - The raw data is available at <a href="http://dx.doi.org/10.1016/j.jtbi.2015.10.040">here</a> and also in the GitHub repository.
    - run ***main.py*** directly.

- ### Real-data experiment 2: Wave Propagation Through Vegetation
    - The raw data is available at <a href="https://doi.org/10.6084/m9.figshare.13026530.v2">here</a>. The preprocessed data is also stored in the GitHub repository.
    - run ***main.py*** directly
