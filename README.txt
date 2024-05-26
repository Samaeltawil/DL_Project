For the Mini-Project of the Deep Learning course, we decided to work on the MMHS150K dataset. This dataset contains 150,000 tweets with images,
and each tweet is labeled as hate speech or not. The goal of this project is to create a model that can predict if a tweet contains hate speech or not.
The dataset is available at https://gombru.github.io/2019/10/09/MMHS/.

To run the project, you need to have the following directory structure:

DL_miniproject
├── dataset/
│   ├── img_resized/
│   │   ├── 1114679353714016256.jpg
│   │   ├── ...
│   │   └── 1110368198786846720.jpg
│   ├── img_tens*/
│   ├── img_txt/
│   │   ├── 1114679353714016256.json
│   │   ├── ...
│   │   └── 1110368198786846720.json
│   ├── test*/
│   │   ├── hate*/
│   │   └── non-hate*/
│   ├── train*/
│   │   ├── hate*/
│   │   └── non-hate*/
│   └── MMHS150K_GT.json
├── results*/
│   ├── plots*/
├── custom_dataset.py
├── custom_models.py
├── main.py
├── main.sh
├── preprocessing.py
├── requirements.txt
├── README.txt
└── utils_preprocessing.py

* Must be created by hand before running the project

To run the project, you need to follow the following steps:

1.  Create a virtual environment and install the required packages by running the following commands:
    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  Download the MMHS150K dataset from the link provided above and extract it in the dataset folder.

3.  Create the img_tens test and train folders in the dataset folder.
    Create the results and plots folders.

4.  Run the preprocessing.py script to create the custom dataset.
    ```
    python preprocessing.py
    ```
    This script will remove all tweets not containing text, and it will preprocess the images to save training time.

5.  Run the main.py script to train the model, evaluate the model and save the results in the results folder.
    ```
    python main.py
    ```
    The script will train the model, evaluate the model and save the results in the results folder.
    The training hyperparameters can be modified in the main.py script.

