# deep-learning-challenge

# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
•	EIN and NAME—Identification columns
•	APPLICATION_TYPE—Alphabet Soup application type
•	AFFILIATION—Affiliated sector of industry
•	CLASSIFICATION—Government organization classification
•	USE_CASE—Use case for funding
•	ORGANIZATION—Organization type
•	STATUS—Active status
•	INCOME_AMT—Income classification
•	SPECIAL_CONSIDERATIONS—Special considerations for application
•	ASK_AMT—Funding amount requested
•	IS_SUCCESSFUL—Was the money used effectively
# Before You Begin
important
The instructions below are now updated to use Google Colab for this assignment instead of Jupyter Notebook. If you have already started this assignment using a Jupyter Notebook then you can continue to use Jupyter instead of Google Colab.
1.	Create a new repository for this project called deep-learning-challenge. Do not add this Challenge to an existing repository.
2.	Clone the new repository to your computer.
3.	Inside your local git repository, create a directory for the Deep Learning Challenge.
4.	Push the above changes to GitHub.
# Instructions
# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1.	Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
•	What variable(s) are the target(s) for your model?
•	What variable(s) are the feature(s) for your model?
2.	Drop the EIN and NAME columns.
3.	Determine the number of unique values for each column.
4.	For columns that have more than 10 unique values, determine the number of data points for each unique value.
5.	Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6.	Use pd.get_dummies() to encode categorical variables.
7.	Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8.	Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.
# Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1.	Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2.	Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3.	Create the first hidden layer and choose an appropriate activation function.
4.	If necessary, add a second hidden layer with an appropriate activation function.
5.	Create an output layer with an appropriate activation function.
6.	Check the structure of the model.
7.	Compile and train the model.
8.	Create a callback that saves the model's weights every five epochs.
9.	Evaluate the model using the test data to determine the loss and accuracy.
10.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.
# Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
•	Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
o	Dropping more or fewer columns.
o	Creating more bins for rare occurrences in columns.
o	Increasing or decreasing the number of values for each bin.
o	Add more neurons to a hidden layer.
o	Add more hidden layers.
o	Use different activation functions for the hidden layers.
o	Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
1.	Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2.	Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
3.	Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4.	Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.
# Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:
1.	Overview of the analysis: Explain the purpose of this analysis.
2.	Results: Using bulleted lists and images to support your answers, address the following questions:
•	Data Preprocessing
o	What variable(s) are the target(s) for your model?
The target variable is the "IS_SUCCESSFUL"
o	What variable(s) are the features for your model?
All columns except for "IS_SUCCESSFUL", "NAME", and "EIN"
o	What variable(s) should be removed from the input data because they are neither targets nor features?
Variables "NAME", and "EIN" were removed, these were not numeric and irrelevant.
•	Compiling, Training, and Evaluating the Model
o	How many neurons, layers, and activation functions did you select for your neural network model, and why?
I chose 3 hidden layers with 12,24 and 36 neurons for no particular reason other than testing since I am new to machine learning. Used 2 activation function, 'RELU' and 'SIGMOID'
o	Were you able to achieve the target model performance?
The target was 75% accuracy which I did not achieve, but the results were close.
o	What steps did you take in your attempts to increase model performance?
I increased the number of hidden layers to 3, also increased the number of epochs to 800 as well as testing with different cutoffs, 250, 500 and 750.
3.	Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
4.	The purpose of this review it to attempt to find a model that would yield a prediction for the highest number of successful applicants for Alphabet soup. Columns “EIN" and "NAME" were removed front the dataset. The Target variable chosen was "IS_SUCCESSFUL" because the name suggests the results are verified, it is also numeric. The rest of the columns (APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS and ASK_AMT) are the feature variables. I tested with different random states and also with different number of epochs. Also changed the hidden layers and values with the below results.
5.	-- Using 2 hidden layers with Relu, PReLU as the Output layer, hidden layers values 12,24,48
268/268 - 1s - loss: 0.6315 - accuracy: 0.7368 - 692ms/epoch - 3ms/step
Loss: 0.6315085291862488, Accuracy: 0.7367929816246033
6.	'''Best results considering loos/accuracy '''
--Using 2 hidden layers with Relu, sigmoid as outer layer, hidden layers values 12,24,48
268/268 - 0s - loss: 0.5572 - accuracy: 0.7368 - 263ms/epoch - 981us/step
Loss: 0.5571669936180115, Accuracy: 0.7367929816246033
7.	-- Using 2 hidden layers with Relu a third hidden layer with LeakyReLU, sigmoid as Output layer, hidden layers values 12,24,48
268/268 - 0s - loss: 0.5608 - accuracy: 0.7388 - 249ms/epoch - 928us/step
Loss: 0.5607810020446777, Accuracy: 0.7387754917144775
8.	-- Using a third hidden layer with LeakyReLU, SoftMax as Output layer, hidden layers values 12,24,48
268/268 - 0s - loss: 0.5609 - accuracy: 0.5349 - 236ms/epoch - 882us/step
Loss: 0.5608648657798767, Accuracy: 0.5349271297454834
9.	'''Best result accuracy only'''
--Using 2 hidden layers with Relu a third hidden layer with LeakyReLU, sigmoid as outer layer, hidden layers values 24,48,96
268/268 - 0s - loss: 0.6040 - accuracy: 0.7392 - 262ms/epoch - 977us/step
Loss: 0.6040363311767578, Accuracy: 0.7392419576644897
10.	-- Using 3 hidden layers with Relu, sigmoid as outer layer, hidden layers values 24,48,96
268/268 - 0s - loss: 0.6273 - accuracy: 0.7366 - 417ms/epoch - 2ms/step
Loss: 0.6273306608200073, Accuracy: 0.7365597486495972
11.	'''Possibly the best config'''
--Using 3 hidden layers with Relu, sigmoid as outer layer, hidden layers values 2, 4, 6
268/268 - 0s - loss: 0.5534 - accuracy: 0.7327 - 430ms/epoch - 2ms/step
Loss: 0.5534052848815918, Accuracy: 0.7327113747596741
12.	I received a warning stating that ".h5" files are considered legacy and the library suggested to save it as ".keras"
Started the project in Jupiter and also uploaded one of the configs to google colab.

# Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.
1.	Download your Colab notebooks to your computer.
2.	Move them into your Deep Learning Challenge directory in your local repository.
3.	Push the added files to GitHub.

