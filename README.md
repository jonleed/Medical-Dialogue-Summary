# Medical-Dialogue-Summary
Welcome to our Project for Deep Learning and Advanced AI. Our project displays a quick use case demo of our trained model through a local FLask application. Our app loads the MTS-Dialog Dataset as well as our pre-trained medical summarization model through HuggingFace Hub.
## Local Usage
### To run the application:
- First run the following command in your directiory [!git clone https://github.com/abachaa/MTS-Dialog.git] 
- Go to [HuggingFace](https://huggingface.co/) and get login/session key in order to load the Flask App localy
- Create a python venv and install the requirements: 
 ```bash
    pip install -r requirements.txt
```
- Activate the venv
 ```bash
    ./venv/Scripts/Activate
```

- Run the the Flask app:
 ```bash
    python ./eval.py
```
- select 'n' for git credential

