from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

class Eigenspectrum:

	# Eigenspectrum is composed of:
	# eigenvalues
	# % info for each eigenvalue
	# you can plot it using show()

	def __init__(self, F):
		self.eigenvalues = eigh(np.cov(F.T),eigvals_only=True)
		total_inf = np.sum(self.eigenvalues)
		self.info = self.eigenvalues/total_inf*100

	def show(self):
		info = self.info[::-1]
		plt.plot(range(0,len(self.eigenvalues)),info)
		plt.title('Eigenspectrum')
		plt.show()

class Preprocessor:

	# Preprocessor is composed of:
	# scaler (minmax or standard)
	# a pc projector
	# you can also remove features using mutual_info_select()

	def __init__(self,scaler_type,n_components):
		if scaler_type == "standard":
			self.scaler = StandardScaler()
		elif scaler_type == "minmax":
			self.scaler = MinMaxScaler()
		self.pca = PCA(n_components)

	def standardize(self,Ftrain,Ftest):
		Ftrain_std = self.scaler.fit_transform(Ftrain)
		Ftest_std = self.scaler.transform(Ftest)
		return (Ftrain_std,Ftest_std)

	def project_on_pc(self,Ftrain,Ftest):
		Ftrain_pca = self.pca.fit_transform(Ftrain)
		Ftest_pca = self.pca.transform(Ftest)
		return (Ftrain_pca,Ftest_pca)

	def mutual_info_select(self,F,y,threshold):
		mi = list(enumerate(mutual_info_classif(F,y)))
		f_best = []
		for (ind,rank) in mi:
			if rank > threshold:
				f_best.append(ind)
		return f_best



