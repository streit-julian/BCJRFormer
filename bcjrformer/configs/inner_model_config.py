from bcjrformer.configs.utils import BooleanFlag


from spock import spock


@spock
class InnerModelConfig:
    # number of sequences in the dataset) - Only supported by some models
    n_sequence_min: int = 1
    n_sequence_max: int = 1

    # Determines how many standard deviations will be used for the window construction of the inner model
    delta_std_multiplier: float = 3.5

    compare_bcjr: BooleanFlag = BooleanFlag.FALSE
