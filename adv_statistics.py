#still working...
#different statistics over low level features

def negentropy(A):
	ne = []
	for u in A:
		a1 = 2
		G1 = 1/a1*np.log(np.cosh(2*u))
		v = np.random.normal(0, 1, len(u))
		G2 = -np.exp(-np.power(v,2)/2)
		ne.append(np.power(np.mean(G1)-np.mean(G2),2))
	return np.array(ne)

#good for discrete features
def differences_entropy(A):
	dH = []
	for f in A:
		Y = np.diff(f)
		P = []
		for yk in Y:
			ntimes = Y.tolist().count(yk)
			P.append(ntimes/len(Y))
		print P
		dH.append(entropy(Y)/np.log(len(Y)))
	return np.array(dH)