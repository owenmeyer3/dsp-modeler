# project folder = .
# input_signal = ./input/frequency=80/file_length_seconds=10/input.wav
# output_signals = ./outputs/distortion=0.14/filter=0.28/output.wav

# creates a .wav file as input_signal for recording different param permutation outputs
def create_input_signal(frequency, file_length_seconds):
    pass

# takes input .wav file and input control state %s, 
# like [{"control":"distortion", "value": 0.14}, {"control":"filter", "value": 0.28}]
# to create output_signals
def create_output_signal(input_signal_file, controls):
    pass
    # Get input signal, pass through device, save recording as output_signal

if __name__ == '__main__':
    mode = 'input' # can be 'input' or 'output'
    sample_rate=96000 # hz

    input_args={
        'frequency':80,
        'file_length_seconds':10,
    }

    output_args={
        'input_signal_file':80,
        'controls': [{"control":"distortion", "value": 0.14}, {"control":"filter", "value": 0.28}]
    }

    if mode=='input':
        create_input_signal(**input_args)
    else:
        create_output_signal(**output_args)
