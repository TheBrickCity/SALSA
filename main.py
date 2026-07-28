import LWE_Sample_Manager, dataset


LWE = LWE_Sample_Manager.LWE(10, 10, 10, 2, 3, 10)
print(LWE.n,LWE.m, LWE.q,LWE.h, LWE.sigma)
LWE.generate()
print("Secret", LWE.s)
print("Matrix A", LWE.A)
print("Error", LWE.e)
print("Vector b", LWE.b)

integer_base = 81 # base that integer b and integers from vecors in A are converted into before tokenizing
dataset = dataset.LWEDataset(LWE, integer_base)
print(dataset._decode_int(dataset._encode_int(9)) == 9)