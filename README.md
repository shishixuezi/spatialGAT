
[version-image]: https://img.shields.io/badge/python-3.8-brightgreen
[version-url]: https://www.python.org/
# SpatialGAT: Spatial Attention Based Grid Representation Learning for Predicting Origin-Destination Flow

[![Python Version][version-image]][version-url]

![Overview](/assets/overview.png)


This is the implementation of the spatialGAT in the paper: **Spatial attention based grid representation learning for predicting origin–destination flow** in **IEEE Big Data 2022**. 
We collected multimodal characteristics of regions, such as road network densities and facility distributions, from several open-source datasets and used them as grid signals, 
and constructed a spatial attention-based deep graph network to generate grid embeddings and used them to predict the OD volumes.

## Dependencies

* numpy
* pandas
* torch 
* torch_geometric
* matplotlib
* scikit-learn
* jismesh
* pyproj

## Prepare environment
```
conda create -n spatialGAT
conda activate spatialGAT
conda install pip
python -m pip install -r requirement.txt
```

After downloading this repository, run:

```
cd spatialGAT
mkdir result
mkdir explain
```

### About Data
- OD data

The OD data used in the paper come from SoftBank Group Corporation, and we are not allowed to open them to the public. 
Currently, we provide a sample file to indicate the format of OD volumes.
Now we are preparing synthetic data in the same scope using [**PseudoPFlow**](https://onlinelibrary.wiley.com/doi/pdf/10.1111/mice.13285) data. 
Once it is finished, we will update them in this repository.

- Region attributes

You can find them in the `data` folder. You can also download the files from URLs provided in the paper.


## To Run Codes

### How to generate the prediction

Run the following code:

```
python main.py
```

### How to generate the explanation

First put the generated `model.pth` file into the `explain` folder. Then run the explanation generation module:

```
python explain.py
```

You can find the diagram of feature important analysis in the `explain` folder.

## Citation

```
@inproceedings{cai2022spatial,
  title={Spatial attention based grid representation learning for predicting origin--destination flow},
  author={Cai, Mingfei and Pang, Yanbo and Sekimoto, Yoshihide},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={485--494},
  year={2022},
  organization={IEEE}
}
```

