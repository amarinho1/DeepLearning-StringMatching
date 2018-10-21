# DeepLearning-StringMatching - M.Sc. Thesis

Approximate String Matching and Duplicate Detection in the Deep Learning Era

Python 2 and Keras 1.2.2
+ `deepneuralnetwork.py` - 2-Stacked BiGRU encoder with various extensions.
+ `feature/deepneuralnetwork_feature.py` - Feature variant of `deepneuralnetwork.py`. Takes one extra feature as input.

Python 3 and Keras 2.2.2
+ `functionaldnn.py` - 2-Stacked BiGRU encoder with various extensions. Functional Keras Model.
+ `stacked_bilstm_max.py`- 3-Stacked RNN encoder with maxpooling between layers.
+ `feature/stacked_bilstm_max_feature.py` - Feature variant of `stacked_bilstm_max.py`. Takes one extra feature as input.

`layers.py` - Custom Keras layers and callbacks.


INSTRUCTIONS

1 - Download and unzip the datasets from the the datasets folder:

The available datasets are:
GeoNames Dataset
JRC-Names Dataset
Historical Place Names

2- Run the desired model followed by the dataset name `python3 desired_model.py dataset_name` or `python desired_model.py dataset_name`.

3- The results will be outputed to the terminal.

