"""
Query and download in a CSV file a Google AI's jobname

acocac@gmail.com
"""

from googleapiclient import discovery
from google.oauth2 import service_account
import pandas as pd
import json

#Set key_path to the path to the service account key
key_path = r"F:\acoca\research\gee\dataset\AMZ\serviceaccount\thesis-240720-13122429d328.json"

# Define the credentials for the service account
credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

# Define the project id and the job id and format it for the api request
project_id_name = 'thesis-240720'
project_id = 'projects/{}'.format(project_id_name)
job_name = 'AMZ_hpt_train_20200123131058'
job_id = '{}/jobs/{}'.format(project_id, job_name)

# Build the service
ml = discovery.build('ml', 'v1', credentials=credentials)

# Execute the request and pass in the job id
request = ml.projects().jobs().get(name=job_id).execute()

# Get just the best hp values
best_model = request['trainingOutput']['trials'][0]
print('Best Hyperparameters:')
print(json.dumps(best_model, indent=4))

# Or put all the results into a df
# Create a list for each field
trial_id, accuracy, steps, epochs, batchsize, learning_rate, convrnn_filters, optimizertype, experiment = [], [], [], [], [], [], [], [], []

train_samples_all = 45312*3
train_samples_sample = 23808*3

train_samples = train_samples_all

# Loop through the json and append the values of each field to the lists
for each in request['trainingOutput']['trials']:
    trial_id.append(each['trialId'])
    accuracy.append(each['finalMetric']['objectiveValue'])
    steps.append(each['finalMetric']['trainingStep'])
    epochs.append(int(each['finalMetric']['trainingStep'])/(train_samples/int(each['hyperparameters']['batchsize'])))
    batchsize.append(each['hyperparameters']['batchsize'])
    learning_rate.append(each['hyperparameters']['learning_rate'])
    convrnn_filters.append(each['hyperparameters']['convrnn_filters'])
    optimizertype.append(each['hyperparameters']['optimizertype'])
    experiment.append(each['hyperparameters']['experiment'])

# Put the lsits into a df, transpose and name the columns
df = pd.DataFrame([trial_id, accuracy, steps, epochs, batchsize, learning_rate, convrnn_filters, optimizertype, experiment]).T
df.columns = ['trial_id', 'accuracy', 'steps', 'epochs', 'batchsize', 'learning_rate', 'convrnn_filters', 'optimizertype', 'experiment']

# Display the df
print(df.head())