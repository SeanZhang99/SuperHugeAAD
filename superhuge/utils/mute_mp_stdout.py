import os
import sys


def mute_mp_stdout():
    sys.stdout = open(os.devnull, "w")
