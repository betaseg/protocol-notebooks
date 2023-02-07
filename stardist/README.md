# 3D segmentation of secretory granules with 3D stardist

![](../figs/granules.png)


The code is this folder demonstrates how to train a Stardist model to segment secretory granules from 3D FIB-SEM data as described in the paper:

*Müller, Andreas, et al. "3D FIB-SEM reconstruction of microtubule–organelle interaction in whole primary mouse β cells." Journal of Cell Biology 220.2 (2021).*


For general question regarding those parameters, please see https://github.com/stardist/stardist.

1. Install tensorflow with gpu support 

2. Install stardist and dependencies:

    - `pip install stardist tqdm`
    - `pip install git+https://github.com/stardist/augmend.git`
    
3. Download the example data (or adapt your own data into the same format)

    - `wget https://syncandshare.desy.de/index.php/s/5SJFRtAckjBg5gx/download/data_granules.zip`
    - `unzip data_granules.zip`

   which should result in the following folder structure:
    ```
    data_granules
    ├── train
    │   ├── images
    │   └── masks
    └── val
        ├── images
        └── masks
    ```

## Usage

Simply run the [notebook](stardist.ipynb) for training a stardist model and applying it on new stacks. 