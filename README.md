# GenerateJokes
## Introduction
The main purpose of this code is to clean the jokes data and fine-tune the GPT2LMHEADmodel to generate jokes. In this training, only 5 epochs of the first 20,000 rows of data were trained due to the large size of the data.

## Getting Started
The raw data is [Short Jokes](https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes/),this data was obtained by the original author by searching from multiple websites that contain funny short jokes.This data was obtained by the original author by searching from multiple websites that contain funny short jokes.
The model used for fine-tuning is the [GPT2LMhead](https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel) model of GPT2, which is used because GPT2 has been trained on a large amount of data with good results, and this version has a Head that can be used to output the results.

##
              
