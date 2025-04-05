## Purpose:
unify preprocessing and data creation methods between different model training (Offline EEGNet, Riemannian Decoder) and model testing (raspy) pipelines

Please run each of the python file to get a grasp of how they work.

* **dataset.py**: contains everything relevant to creating a h5 dataset. The main function is create_dataset(config, h5_path)
* **preprocessor.py**: contains everything that has to do with preprocessing a piece of EEG data
* **utils.py**: contains standalone functions helpful to data reading in general
