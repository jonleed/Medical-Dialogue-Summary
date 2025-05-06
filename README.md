# Medical-Dialogue-Summary
Welcome to our Project for Deep Learning and Advanced AI. Our project displays a quick use case demo of our trained model through a local FLask application. Our app loads the MTS-Dialog Dataset as well as our pre-trained medical summarization model through HuggingFace Hub.
## Local Usage
### To run the application:
- First run the following command in your directiory [!git clone https://github.com/abachaa/MTS-Dialog.git] 
- Go to [HuggingFace](https://huggingface.co/), get login/session key and keep it ready (needed to load the model) 
- Create a python virtual environment:
- Activate the venv:
 ```bash
    ./venv/Scripts/Activate
```
Install the requirements: 
 ```bash
    pip install -r requirements.txt
```
- Run the the Flask app:
 ```bash
    python ./eval.py
```
- When protmpted, paste your HuggingFace Hub login/session key
- Enter 'n' for git credential

### Now Scroll through the table and click the summarize button to summarize each dialogue instance!

