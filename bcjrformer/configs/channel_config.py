from spock import spock
from spock.utils import ge


@spock
class IDSChannelConfig:
    p_i: float
    p_d: float
    p_s: float

    fixed_del_count: int | None = None

    def __post_hook__(self):
        if self.fixed_del_count is not None:
            ge(self.fixed_del_count, 0)

            if self.p_i > 0:
                raise NotImplementedError(
                    "Fixed deletion count is not supported for non-zero insertion probability"
                )
