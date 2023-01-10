from models import CPPN

model = CPPN(input_vector_length=12, num_nodes=16, num_layers=9, output_vector_length=3, positional_encoding_bins=12)
model = model.to('mps')