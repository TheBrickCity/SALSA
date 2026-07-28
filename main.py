import LWE_Sample_Manager, dataset


LWE = LWE_Sample_Manager.LWE(10, 10, 10, 2, 3, 10)
print(LWE.n,LWE.m, LWE.q,LWE.h, LWE.sigma)
LWE.generate()
print("Secret", LWE.s)
print("Matrix A", LWE.A)
print("Error", LWE.e)
print("Vector b", LWE.b)


dataset = dataset.LWEDataset(
    LWE.A,
    LWE.b,
    81 # base for tokens
)

A, b = dataset[0]
print(dataset.__len__())