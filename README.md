# Medical-Dialogue-Summary
Welcome to our Project for Deep Learning and Advanced AI. Our project displays a quick use case demo of our trained model through a local FLask application. Our app loads the MTS-Dialog Dataset as well as our pre-trained medical summarization model through HuggingFace Hub.
## Notebook File
To train the model yourself, run the notebook file. If you want to upload the model to your own account, edit the MODEL_PATH variable to your own repository Path. If done, additionally edit the path in the "eval.py" file to pull from your new repository.

## Local Usage - Flask App
### To run the application:
- First run the following command in your directiory:
 ```bash
!git clone https://github.com/abachaa/MTS-Dialog.git
```
- Go to [HuggingFace](https://huggingface.co/), get login/session key and keep it ready (needed to load the model) 
- Create a python virtual environment:
```bash
    python -m venv <environment_name>
```
- Activate the venv and install the requirements: 
 ```bash
    pip install -r requirements.txt
```
- Run the the Flask app:
 ```bash
    python ./eval.py
```
- When protmpted, paste your HuggingFace Hub login/session key
- Enter 'n' for git credential
    - (You may be asked twice)
- You are now running the app locally. On your browser, go to [http://127.0.0.1:5000/]
### Now Scroll through the table and click the summarize button to summarize each dialogue instance!

