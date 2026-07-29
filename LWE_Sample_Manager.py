import numpy as np

class LWE:
    def __init__(self, n, m, q, h, sigma, gaussian_bound):
        self.n = n
        self.m = m
        self.q = q
        self.h = h
        self.sigma = sigma
        self.gaussian_bound = gaussian_bound
        self.s = None
        self.A = None
        self.e = None
        self.b = None

        self.create_gaussian_table()

    def generate(self):
        self._generate_secret()
        self._generate_matrix()
        self._generate_error()
        self._generate_b()

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

    def _generate_error(self):
        e = np.array([
                self.sample_discrete_gaussian()
                for _ in range(self.m)
                ],dtype=np.int64)
        self.e = e
        return e

    def create_gaussian_table(self): # creates table of values and their probabilities up to a bounded multiple of the standard deviation
        self.gaussian_values = np.arange(
            -self.gaussian_bound,
            self.gaussian_bound + 1
        )
        weights = np.exp(
            -(self.gaussian_values ** 2)
            /
            (2 * self.sigma ** 2)
        )
        self.gaussian_probabilities = (
                weights / np.sum(weights)
        )

    def sample_discrete_gaussian(self): # returns one value from the gaussian probabilities
        return np.random.choice(
            self.gaussian_values,
            p=self.gaussian_probabilities
        )

    def _generate_b(self):
        b = (self.A @ self.s + self.e) % self.q
        self.b = b
        return b