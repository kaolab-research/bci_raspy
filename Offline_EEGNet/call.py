from instantiate import instantiate_model


path = '/data/raspy/trained_models/wombats_dermatology_EEGNet_2023-07-22_S1_OL_1_RL/'
config_file = path + 'config.yaml'
model_architecture = path + 'EEGNet.py'
model_file = path + '0'

# Instantiate the model using the configurations from 'config.yaml'
model = instantiate_model(config_file, model_architecture, model_file)

y = model.eval()  # Replace with the actual method to make predictions in your EEGNet class
