import numpy as np

class CloudState:
    def __init__(self, vel):
        self.vel = vel # [y,x]
        # could add spreading parameter
