# Readme before creating a new decoder module

## Formatting
1. The decoder module must have an attribute 'name' containing the name of the Decoder class
1. The Decoder class must inherit from torch.nn.Module
1. The Decoder instance must have an attribute 'input_length' which is a positive integer
1. The Decoder instance must have an attribute 'input_dim' which is a positive integer
1. Should be able to handle variable shape input, i.e. w/ or w/o a batch dimension, w/ or w/o a length dimension for input_length=1
## Inputting different kwargs:
Write to appropriate YAML file under params->decoder_kwargs
