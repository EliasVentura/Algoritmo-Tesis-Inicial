#Algoritmo para realizar clasificacion. Creado por Elias Ventura-Molina

import numpy as np

class AlfaBeta:
	def __init__( self, ds ):
		self.ds = ds

	def f(self, x):
		return 4 * x - 2

	def alpha( self, a, b ):
		Amin = np.min(np.min(self.ds[:, 0:-1]))
		Amax = np.max(np.max(self.ds[:, 0:-1]))

		a0 = self.f(Amax) - self.f(Amin)
		return 4*(a-b) + a0

	def beta( self, a, b ):
		Amin = np.min(np.min(self.ds[:, 0:-1]))
		Amax = np.max(np.max(self.ds[:, 0:-1]))

		a0 = self.f(Amax) - self.f(Amin)
		res = a/4.0 + b - a0

		return Amin if res <= Amin else Amax if res >= Amax else res

	def y( self, xM, yM ):
		zM = np.zeros( (yM.size, xM.size) )
		for i in range(yM.size):
			for j in range(xM.size):
				zM[i,j] = self.alpha( yM[i], xM[j] )
		return zM;

	def z( self, M, xM ):
		rows = M.shape[0]
		cols = M.shape[1] # -1 for class

		result = []

		for i in range(rows):
			vals = []
			for k in range(cols):
				vals += [self.beta( M[i, k], xM[k] )]

			result += [min(vals)]

		return np.array(result);

	def one_hot(self, m, k):
		onehot = np.zeros(m)
		onehot[k] = 1
		return onehot

	def train( self ):
		#m = np.unique( self.ds[:, -1] ).size
		n = np.size( self.ds, axis=1 ) - 1
		m = np.size( self.ds, axis=0 )

		self.M = np.full( (n, n), -np.inf )
		for i, p in enumerate(self.ds):
			xM = p[0:-1]
			yM = xM #self.one_hot(m, i)
			zM = self.y( xM, yM )
			self.M = np.maximum( zM, self.M )

	def classify( self, p ):
		print self.z( self.M, p )
		return np.argmax( self.z( self.M, p ) )
