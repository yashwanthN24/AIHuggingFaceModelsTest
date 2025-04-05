from transformers import pipeline
# Transformers package makes extremmely east to access these free hugging face models
# and  to download and use the model in your computer 

# langchain is a package that allows you to run multiple models in a sequence 
#  helps working with llms easy and fast

"""
    First Create a Virtual Environment 

    For Windows:
        python -m venv env
    For Mac:
        python3 -m venv env

    To activate the virtual environment:
        For Windows:
            ./env/Scripts/activate
        For Mac:
            source ./env/bin/activate


    
    touch requirements.txt

    touch main.py

    pip install -r requirements.txt

    python main.py


    Create a Hugging Face Account and create a token

    IN CLI :

    huggingface-cli login

    Then enter the token you copied from hugging face website
    Then you can use the models from hugging face



"""


# Pipeline is a simplified way of running various models


# For example, if you want to use a summarization model, you can do it like this:

# This line downloadds the model and caches it in your local directory
model = pipeline( "summarization" , model="facebook/bart-large-cnn")

# Then you can use the model like this and give relevant text to the model
# The model will return a summary of the text you provided
response = model("text to summerize")

# The response will be a dictionary with the summary
print(response)