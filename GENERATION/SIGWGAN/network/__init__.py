from .generators import LSTMGenerator, LogSigRNNGenerator

GENERATORS = {'LSTM': LSTMGenerator, 'LogSigRNN': LogSigRNNGenerator}


def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)
