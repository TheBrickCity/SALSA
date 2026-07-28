import LWE_Sample_Manager

LWE = LWE_Sample_Manager.LWE(10, 10, 10, 2, 3)
print(LWE.n,LWE.m, LWE.q,LWE.h, LWE.sigma)
LWE.generate()
print(LWE.s)
print(LWE.A)