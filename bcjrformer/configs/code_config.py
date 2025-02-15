from enum import Enum, StrEnum
from typing import List
from spock import spock
from bcjrformer.codes.linear_code import CodeFileExtension, LinearCodeType


class InnerCodeType(StrEnum):
    CONVOLUTIONAL = "convolutional"
    MARKER = "marker"


@spock
class ConvolutionalCodeConfig:
    """
    Convolutional Code Configuration

    Attributes:
        k: The number of information bits
        g: The generator polynomials
        p: The number of bits to encode the alphabet
        random_offset: Whether to randomly offset codeword
    """

    k: int
    g: List[int]
    p: int
    random_offset: bool = True


@spock
class MarkerCodeConfig:
    """Marker Code Configuration

    Attributes:
        marker: The marker sequence
        N_c: Number of bits between two markers
        p: Number of bits to encode the alphabet
    """

    # the marker sequence
    marker: List[int]

    # number of bits between two markers
    N_c: int

    # number of bits to encode the alphabet
    p: int = 1


class InnerCodeConfig(Enum):
    CONVOLUTIONAL = ConvolutionalCodeConfig
    MARKER = MarkerCodeConfig


@spock
class LinearCodeConfig:
    """Linear Code Configuration

    Attributes:
        code_type: Type of code
        k: Linear code k
        n: Linear code n
        q: Alphabet/Field size
        custom_file_name: Custom file name of the code's pc matrix
        file_extension: File extension of the code's pc matrix
    """

    code_type: LinearCodeType = LinearCodeType.LDPC
    k: int
    n: int
    q: int = 2
    custom_file_name: str | None = None
    random_weights: bool = False
    file_extension: CodeFileExtension | None = None


@spock
class ConcatenatedCodeConfig:
    """Either an outer code or a combination of inner and outer codes"""

    outer_code_config: LinearCodeConfig
    inner_code_config: InnerCodeConfig | None = None


InnerCodeConfigType = ConvolutionalCodeConfig | MarkerCodeConfig


def code_config_to_code_type(inner_code_config: InnerCodeConfigType) -> InnerCodeType:
    if isinstance(inner_code_config, ConvolutionalCodeConfig):
        return InnerCodeType.CONVOLUTIONAL
    elif isinstance(inner_code_config, MarkerCodeConfig):
        return InnerCodeType.MARKER
    else:
        raise ValueError(f"Invalid inner code config: {inner_code_config}")
