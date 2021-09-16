# EDM-FDE

This repository contains code for Euclidean Distance Matrix-based Fault Detection and Exclussion (FDE) based on the paper "Euclidean Distance Matrix-based Rapid Fault Detection and Exclusion" from the ION GNSS+ 2021 conference.

## Install

This repository was developed and tested with Python 3.8+

Install python requirements with `pip install -r requirements.txt`

## Run Instructions

EDM-based FDE algorithm is implemented in [src/edm_fde.py](https://github.com/betaBison/edm-fde/blob/main/src/edm_fde.py).

ION GNSS+ presentation/paper figures can be replicated from previously logged data with `python ion_figures.py`

Chemnitz results can be replicated (up to the random initialization) with `python main_chemnitz.py`

## Citation
If referencing EDM-based FDE in your work, please cite the following paper:
```
@inproceedings{Knowles2021,
author = {Knowles, Derek and Gao, Grace},
title = {{Euclidean Distance Matrix-based Rapid Fault Detection and Exclusion}},
booktitle = {Proceedings of the 34th International Technical Meeting of the Satellite Divison of the Institute of Navigation, ION GNSS + 2021},
publisher = {Institute of Navigation},
year = {2021}
}
```
