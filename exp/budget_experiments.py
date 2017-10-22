import fire
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenstien


def foward():
    """Optimize using voltage budget."""
    pass


def reverse():
    """Optimize using ISI."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse})
