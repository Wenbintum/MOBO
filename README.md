# Discovering high entropy alloy electrocatalysts in vast composition spaces with multi-objective optimization

## Overview
This repository houses all the code and data that should allow for the reproduction of the results in the paper: 'Discovering high entropy alloy electrocatalysts in vast composition spaces with multi-objective optimization'.

- The DFT dataset comprises the following: The in-domain HEA dataset contains subsets such as AgIrPdPtRu, AuOsPdPtRu, and CuPtReRhRu; the out-of-domain HEA datasets include subsets for composition-diversity and component-diversity. All datasets have been prepared for training the GNN model. For a detailed description of these datasets, please refer to our paper.
- The source code for the multi-objective framework, encompassing both multi-objective optimization and the GNN model used to discover HEA electrocatalysts, are available in directory "code_MOBO_HEAs" together with a tutorial.

## Citation
If you employ this codebase in your research, please acknowledge it by citing:
```latex
@article{xu_mobo_hea_2023,
title = {Discovering high entropy alloy electrocatalysts in vast composition spaces with multi-objective optimization},
author = {Wenbin Xu, Elias Diesen, Tianwei He, Karsten Reuter and Johannes T. Margraf},
journal = {in preparation},
year = {2023},
}
```
