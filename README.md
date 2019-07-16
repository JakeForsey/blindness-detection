# APTOS 2019 Blindness Detection

## Setup

Add APTOS 2019 Blindness Detection dataset to `data/aptos2019-blindness-detection` folder at root of project.

```
virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
#... when done
deactivate
```


## Training

```
./train.py
```

## Submitting

TODO validate this approach and update 

Inspired by: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/82092

#### Setting up repo
1. Download the git repo as a zip file
2. Upload the git repo to kaggle kernel as a dataset

#### Setting up the trained model
1. Upload the trained checkpoint file to kaggle kernel as a dataset

#### Run
1. Copy and paste the following into the kernel
```python
import sys

sys.path.append("../input/path/to/root/of/blindness-detection/")

from submit import main

main("<path_to_checkpoint>", "<path_to_image_directory", "path_to_image_data_frame")
```

2. Update "<path_to_checkpoint>", "<path_to_image_directory", "path_to_image_data_frame"
3. Run kernel