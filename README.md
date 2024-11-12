# QAModel-B2

# GROUP 1:

The uploaded files on Git are: 

1. Original DataSet:  "COVID-QA_Dataset.csv"
2. Preprocessed and new dataset:   "formatted_output.csv"
3. Python file:   "Practicum_B2.ipynb"


Tasks Performed:

1. The original dataset included two unnecessary columns which were dropped and only two columns question and answer were used.
2. After which a format for the LLM output was decided and a new dataset in that format was created in the name "formatted_output.csv" 
3. The output of the final LLM response should be of the similar format shown in the csv file "formatted_output.csv" in the form:
        Question: ............
        Answer: ..............

4. Tags were generated for the new dataset "formatted_output.csv" and divided into four clusters and according to which each pair of question and answer were classified and given tag number based on the cluster numbers. Use these tags to build the model architecture and perform the further steps on the updated dataset "formatted_output.csv"


Expected results:

The final LLM model should be given a question as a prompt related to covid and the output should be an answer for the same. If the question is not related to Covid then the answer should be that "please ask a covid related question" or "I am not able to answer questions outside of the covid domain" or any similar type of response.


Format for the generated output: 

### Question: What is covid-19?
### I am a covid Q&A Agent... 

### Answer: "Answer should be printed related to the question"



## Group 2

### Files on Git:
- Group2.py

### Tasks Performed:

1. Performed tokenization and created a vocabulary.
2. Developed a basic structure of a Transformer model.
3. Used a Naive Bayes Classifier for text classification.


## Group 3

### Tasks Performed:

1. **Data Preparation**:
   - Reading and preprocessing the data (CSV file)
   - Extracting question and answer pairs
   - Splitting data into training and testing sets

2. **Model Selection and Training**:
   - Using a pre-trained BERT model for sequence classification
   - Creating a custom dataset class for the specific task
   - Training the model using an appropriate optimizer (AdamW) and loss function

3. **Training Loop**:
   - Iterating over batches
   - Forward pass, loss calculation, backpropagation, and weight updates

### Group 4
### Tasks Performed

**Training Loop:**
Looped over the specified number of epochs.
For each epoch:
Train the Model: Called the train_epoch function to perform forward and backward propagation, updating model weights based on training data.
Evaluate the Model: Called the eval_model function to evaluate performance on the validation set.
Print Losses: Displayed the training and validation loss after each epoch for monitoring model improvement.

**Save Model**
Save Model Checkpoint: Saved the trained model parameters to a directory (model_checkpoint) for later use or further fine-tuning.
Save Tokenizer: Saved the tokenizer to the same directory, ensuring compatibility with the saved model for future predictions.
