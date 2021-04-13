# Flow classifier
A website fingerprinting attack method based on Deep learning, implemented with pytorch.
Project for `Advanced technology of computer network` course.

## Requirements
* python>=3.6
* pytorch>=1.6.0
* pyhocon
* numpy

## Usage
* Download and extract the dataset to the project root.
* Modify the configuration name in `src/train.py` to specify the experiment to run. Predefined experiment configurations could be found in `src/experiments.conf`.
* Execute `python -m src.train` at the project root.
    * It might take a while to cache the datasets the first time you run it.
    * It would be very slow if run on cpu. 
