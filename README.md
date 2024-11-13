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

## Group 5

### Tasks Performed:

### **Streamlit Frontend for Q&A Model**

We have built a simple Streamlit application to interact with the fine-tuned BERT model for Question-Answering. The Streamlit frontend allows users to enter a question, which is then processed by the model to generate an answer.

**Key Steps**:

- **Model and Tokenizer Loading**: The Streamlit app loads the trained BERT model and tokenizer from a local directory (./model_checkpoint).

- **Prediction Function**: The app defines a function get_prediction() that tokenizes the user-input question, feeds it into the model, and retrieves the model's predicted answer along with a confidence score.

- **Streamlit Layout**: The user interface includes:
An input field for entering a question.
A button to submit the question and generate an answer.
Output fields that display the answer and model confidence score.


Here are the tasks performed in *Step 7* and *Step 8*:

### Step 7: Training and Evaluation
1. *Set Device*: Specify the device (GPU or CPU) for running the model and move the model to the chosen device.
2. *Training Loop*:
   - Loop over the specified number of epochs.
   - For each epoch:
     - *Train the Model*: Call the train_epoch function to perform forward and backward propagation, updating model weights based on training data.
     - *Evaluate the Model*: Call the eval_model function to evaluate performance on the validation set.
     - *Print Losses*: Display the training and validation loss after each epoch for monitoring model improvement.

### Step 8: Save Model
1. *Save Model Checkpoint*: Save the trained model parameters to a directory (model_checkpoint) for later use or further fine-tuning.
2. *Save Tokenizer*: Save the tokenizer to the same directory, ensuring compatibility with the saved model for future predictions.
