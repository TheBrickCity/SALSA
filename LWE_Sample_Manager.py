import numpy as np

class LWE:
    def __init__(self, n, m, q, h, sigma):
        self.n = n
        self.m = m
        self.q = q
        self.h = h
        self.sigma = 3
        self.s = None
        self.A = None
        self.e = None
        self.b = None

    def generate(self):
        self._generate_secret()
        self._generate_matrix()

    def _generate_secret(self):
        s = np.zeros(self.n, dtype=np.int64) # generates 0 array
        indices = np.random.choice(         # chooses h unique indicies from 0 to n-1
            self.n, self.h, replace=False
        )
        s[indices] = 1                      # sets those indicies to 1
        self.s = s
        return s

    def _generate_matrix(self):
        A = np.random.randint(0,self.q, size=(self.m,self.n),dtype=np.int64)
        self.A = A
        return A