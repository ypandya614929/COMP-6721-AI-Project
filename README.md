# COMP-6721-AI-Project
COMP-6721-AI-Project

## To setup virtualenv if not already if running as .py file
    
    virtualenv -p python3.6 env
    source env/bin/activate
    pip install -r requirements.txt

## To run ipynb file, please use jupyter notebook/google colab
    
    The project is build with multiple libraries which is allowed as a part of implementation.
    If using google colab these libraries are already installed, if using jupyter one may
    need to install it using pip/conda.
    
- **Dataset** folder contains all the images for training and testing dataset.
- Dataset/**ai-project-1** which is **pre-trained model**.
- Dataset/**processed_data** which is **pre-processed dataset** as **npy** file.

## Google Colab Execution

Please send an email at yashp6149@gmail.com to get execution environment access and click [here](https://colab.research.google.com/drive/1zqfS3bsUsyRtrd5FbzsXVMOLYhDcDM4Q#scrollTo=6Rdb5-xERu8u) to go to the execution environment.

## Run Program

If the value of **is_data_preloaded** is **True** which means **program reads pre-loaded data** from the directory, otherwise it generates
the data and loads it. While the **True** value of **is_model_saved** parameter indicates that
program reads **pre-trained model** from the directory which is pre stored and if the parameter is set as 
**False** then it **creates new model, trains the model** and then it becomes available for further use such as in evaluation process.

**run_program(is_data_preloaded=True, is_model_saved=True)**

### Note: 
**Please follow comments at import level and at the execution function related to local execution of the program before proceeding.**
