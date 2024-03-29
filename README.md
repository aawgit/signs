# Sinhala Fingerspelling Sign Language Recognition with Computer Vision

A computer vision based desktop application for translating Sinhala Fingerspelling Sign Language Alphabet.

## Live demo (Web app version)
https://signsl.web.app/

## Data
https://www.kaggle.com/datasets/akalankaweerasooriya/sinhala-fingerspelling

Alternative download: https://u.pcloud.link/publink/show?code=XZjE7jVZLgi7N85ivjhHAskliw7IyR8qc5tX


## Installation
1. Clone the repo
2. Download the dataset from a link in the above section.
3. Extract the data (9 directories) to signs/data directory
4. (Optional) Create a virtual environment and activate it
5. Install the requirements by running `pip install -r requirements.txt`

## Running
1. Add the root directory (signs) to PYTHONPATH
- To do that in Linux based systems, run `export PYTHONPATH=$PYTHONPATH:</path/to/signs/directory>`
2. To train the model and run real time translation using a webcam, run src/main.py

## Paper
- https://ieeexplore.ieee.org/document/9906281
- https://www.researchgate.net/publication/364120463_Sinhala_Fingerspelling_Sign_Language_Recognition_with_Computer_Vision

## Web app code base
https://github.com/aawgit/signs-web

