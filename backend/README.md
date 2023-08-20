# Backend

This is the backend for the project using FastAPI. It is simple API containing a single endpoint that returns a classification for a given email.

## Project Setup
please ensure that pip is installed in the system along with the latest version of python or python3

### Install dependencies

```sh
pip install -r requirements.txt
```

### Download the model
As the model is too large in size download the model from the link below and paste this model directory inside the backend folder
[Link] :- https://drive.google.com/drive/folders/1aDBNMtKP1PRlChkKMNE0s5nfdEPXHkmA?usp=drive_link


### Run the server

```sh
uvicorn app.main:app --reload
```

### Train the model


```sh
python ./app/train.py
```

### Running AI interface on gradio

Steps for running the AI-Kart Interface

1> Download the model files that are available in the presentation or from here
[Link] :- https://drive.google.com/drive/folders/1aDBNMtKP1PRlChkKMNE0s5nfdEPXHkmA 

2> Change the 'model1_path' to the local path where the previous downloaded model is stored.

3> Run this IPYNB note book on Google Colab (preferable) or Jupyter Notebook and the AI kart interface will automaticall be generated.


