from googleapiclient import discovery
from google.oauth2 import service_account
import pandas as pd
import json

#Set key_path to the path to the service account key
key_path = "service_account.json"

# Define the credentials for the service account
credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

# Define the project id and the job id and format it for the api request
profect_id_name = 'tutorials-201907'
project_id = 'projects/{}'.format(profect_id_name)
job_name = 'tile_0_563_hpt_train_20200102173533'
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
trial_id, accuracy, batchsize, learning_rate, convrnn_filters, convrnn_layers = [], [], [], [], [], []

# Loop through the json and append the values of each field to the lists
for each in request['trainingOutput']['trials']:
    trial_id.append(each['trialId'])
    accuracy.append(each['finalMetric']['objectiveValue'])
    batchsize.append(each['hyperparameters']['batchsize'])
    learning_rate.append(each['hyperparameters']['learning_rate'])
    convrnn_filters.append(each['hyperparameters']['convrnn_filters'])
    convrnn_layers.append(each['hyperparameters']['convrnn_layers'])

# Put the lsits into a df, transpose and name the columns
df = pd.DataFrame([trial_id, accuracy, batchsize, learning_rate, convrnn_filters, convrnn_layers]).T
df.columns = ['trial_id', 'accuracy', 'batchsize', 'learning_rate', 'convrnn_filters', 'convrnn_layers']

# Display the df
print(df.head())