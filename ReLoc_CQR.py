import copy
import torch
import math
import numpy as np
import scipy.special
import scipy.spatial
from numpy.random import rand
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from torch.autograd import Variable
#import quant_np as qnp

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Loc_CQR():
	'''
	The CQR class implements Conformalized Quantile Regression, i.e. it applies CP to a QR
	Inputs: 
		- Xc, Yc: the calibration set
		- trained_qr_model: pre-trained quantile regressor
		- quantiles: the quantiles used to train the quantile regressor
		- test_hist_size, cal_hist_size: number of observations per point in the test and calibration set respectively
		- comb_flag = False: performs normal CQR over a single property 
		- comb_flag = True: combine the prediction intervals of the CQR of two properties
	'''
	def __init__(self, Xc, Yc, trained_qr_model, type_local = 'gauss', eps= 0.1, knn = 100, test_hist_size = 200, cal_hist_size = 50, quantiles = [0.05, 0.95], comb_flag = False,plots_path=''):
		super(Loc_CQR, self).__init__()

		self.Yc = Yc
		self.Xc = Xc
		
		
		self.type_local = type_local # {'gauss', 'knn'}
		self.eps = eps # variance of the gaussian localizers
		self.knn = knn # nb of neighbors in the knn localizer
		self.trained_qr_model = trained_qr_model
		self.q = len(Yc) # number of points in the calibration set
		self.n = len(Xc) # number of states
		self.test_hist_size = test_hist_size
		self.cal_hist_size = cal_hist_size
		self.quantiles = quantiles
		self.epsilon = 2*quantiles[0]
		self.alpha = 1-self.epsilon
		self.grid_alphas = np.linspace(self.alpha, 1, 101)
		self.Q = (1-self.epsilon)*(1+1/(self.q+1))


		
		self.M = len(quantiles) # number of quantiles
		self.col_list = ['yellow', 'orange', 'red', 'orange', 'yellow']
		self.comb_flag = comb_flag # default: False
		self.plots_path = plots_path

	def weighted_quantile(self, values, quantiles, weights=None, values_sorted=False):
	    values = np.array(values)
	    quantiles = np.array(quantiles)
	    if weights is None:
	        weights = np.ones(len(values))
	    weights = np.array(weights)
	    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
	        'quantiles should be in [0, 1]'

	    if not values_sorted:
	        sorter = np.argsort(values)
	        values = values[sorter]
	        weights = weights[sorter]

	    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
	    weighted_quantiles /= np.sum(weights)
	    
	    return np.interp(quantiles, weighted_quantiles, values)
	

	def get_pred_interval(self, inputs):
		'''
		Apply the trained QR to inputs and returns the QR prediction interval
		'''
		
		if not self.comb_flag:
			return self.trained_qr_model(Variable(FloatTensor(inputs))).cpu().detach().numpy()
		else:
			interval1 = self.trained_qr_model[0](Variable(FloatTensor(inputs))).cpu().detach().numpy() 
			interval2 = self.trained_qr_model[1](Variable(FloatTensor(inputs))).cpu().detach().numpy() 
			pis = []
			for i in range(inputs.shape[0]):
				# LB = min of the lbs 
				# UB = min of the ubs
				pis.append([min(interval1[i][0],interval2[i][0]), min(interval1[i][-1],interval2[i][-1])]) 

			return np.array(pis) 
		


	def get_calibr_nonconformity_scores(self, y, pred_interval, sorting = True):
		'''
		Compute the nonconformity scores over the calibration set
		if sorting = True the returned scores are ordered in a descending order
		'''
		ncm = np.empty(self.q+1)
		ncm[-1] = math.inf
		c = 0
		for i in range(self.n):
			for j in range(self.cal_hist_size):
			
				ncm[c] = max(pred_interval[i,0]-y[c], y[c]-pred_interval[i,-1]) # pred_interval[i,0] = q_lo(x), pred_interval[i,1] = q_hi(x)
				c += 1
		ncm_sort = np.sort(ncm)[::-1] # descending order
		self.tau = np.quantile(ncm_sort, self.Q)

		if False:
			fig = plt.figure()
			plt.scatter(np.arange(len(ncm)), ncm, color='g')
			plt.scatter(self.epsilon*(1+1/self.q)*len(ncm), self.tau,color='r')
			plt.title('calibr ncm')
			plt.tight_layout()
			plt.savefig(self.plots_path+'/calibr_nonconf_scores.png')
			plt.close()

		self.V = ncm[:-1]
		self.compute_H_matrix()



		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
			

		return ncm

	def compute_H_matrix(self):

		
		self.H = np.empty((self.n,self.n))

		for i in range(self.n):
			for j in range(self.n):
				if self.type_local == 'gauss':
					self.H[i,j] = self.compute_gauss_likelihood(self.Xc[i],self.Xc[j])
				else:
					self.H[i,j] = np.linalg.norm(self.Xc[i]-self.Xc[j])



	def get_calibr_scores(self):

		self.calibr_pred = self.get_pred_interval(self.Xc)
		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores(self.Yc, self.calibr_pred, sorting=False)
	
	def compute_gauss_likelihood(self, x_star, xi):

		
		k = len(x_star)
		inv_eps = 1/self.eps
		inv_sigma = inv_eps*np.eye(k)
		
		diff = xi-x_star
		
		ratio = (np.exp(-np.dot(diff.reshape(1,k),np.dot(inv_sigma,diff))))

		return ratio[0]



	def get_H_star(self, x_star):

		
		HH = np.empty((self.n+1,self.n+1))
		HH[:self.n,:self.n] = self.H
		for i in range(self.n):
			if self.type_local == 'gauss':
				HH[i,-1] = self.compute_gauss_likelihood(self.Xc[i],x_star)
				HH[-1,i] = self.compute_gauss_likelihood(x_star, self.Xc[i])
			else:
				HH[i,-1] = np.linalg.norm(self.Xc[i]-x_star)
				HH[-1,i] = np.linalg.norm(x_star-self.Xc[i])


		if self.type_local == 'gauss':
			HH[-1,-1] = self.compute_gauss_likelihood(x_star,x_star)	
		else:
			HH[-1,-1] = 0

		return HH

	def get_P_star(self, H_star):


		Pstar = np.zeros((self.n+1,self.q+1))
		if self.type_local == 'gauss':
			for ii in range(self.q+1):
				Pstar[:,ii] = H_star[:,ii//self.cal_hist_size]
			#for jj in range(self.n+1):
			#	Pstar[jj] = Pstar[jj]/np.sum(Pstar[jj])
		else:
			Pstar = np.zeros((self.n+1,self.q+1))
			for i in range(self.n+1):
				sort_index = np.argsort(H_star[i]) # increasing sorting
				
				r = np.zeros(self.n+1)
				r[sort_index[:self.knn]] = 1
				r_rep = np.hstack((np.repeat(r[:-1], self.cal_hist_size), [r[-1]]))

				Pstar[i] = r_rep

		return Pstar

	def get_scores_threshold(self,x_star):
		'''
		This method extract the threshold value tau (the quantile at level epsilon) from the 
		calibration nonconformity scores (computed by the 'self.get_calibr_nonconformity_scores' method)
		'''

		# independent on alpha_star
		Hstar = self.get_H_star(x_star)
		Pstar = self.get_P_star(Hstar)

		V0 = np.hstack((self.V, [0]))

		#print("=== x star = ", x_star)
		for alpha_tilde in self.grid_alphas:
			
			v_star =  self.weighted_quantile(self.calibr_scores, alpha_tilde, weights = Pstar[-1])
			#print("alpha tilde = ", alpha_tilde, "v_star = ", v_star)
			v1_star = np.empty(self.q)
			v2_star = np.empty(self.q)
			for i in range(self.n):
				Vs = np.hstack((self.V, [v_star]))
				qi1 = self.weighted_quantile(Vs, alpha_tilde, weights=Pstar[i])
				qi2 = self.weighted_quantile(V0, alpha_tilde, weights=Pstar[i])
				#print('Qis=', qi1, qi2)
				v1_star[i*(self.cal_hist_size):(i+1)*(self.cal_hist_size)] = qi1
				v2_star[i*(self.cal_hist_size):(i+1)*(self.cal_hist_size)] =  qi2

			r1 = np.sum(((self.V-v1_star)<=0))/(self.q+1)
			r2 = np.sum(((self.V-v2_star)<=0))/(self.q+1)

			#print("rs = ", r1, r2)
			
			if v_star == math.inf or (r1 >= self.alpha and r2 >= self.alpha):
				#print("----> alpha tilde = ", alpha_tilde, "tau_j = ", v_star)
				return v_star





	def get_cpi(self, inputs, pi_flag = False):
		'''
		Returns the conformalized prediction interval (cpi) by enlarging the 
		QR prediction interval by adding an subtracting tau from the lower and upper bound resp.
		'''
		pi = self.get_pred_interval(inputs)

		cpi = np.vstack((pi[:,0],pi[:,-1])).T
		cpi[:,0] -= self.tau
		cpi[:,-1] += self.tau

		loc_cpi = np.empty((pi.shape[0],2))

		for j in range(pi.shape[0]): 
			
			tau_j = self.get_scores_threshold(inputs[j])

			loc_cpi[j] = [pi[j,0]-tau_j,pi[j,-1]+tau_j]
			
		
		if pi_flag:
			return cpi, loc_cpi, pi
		else:
			return cpi, loc_cpi

	def get_coverage_efficiency(self, y_test, test_pred_interval):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		
		c = 0
		for i in range(n_points):
			for j in range(self.test_hist_size):
				if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
					c += 1
		coverage = c/(n_points*self.test_hist_size)

		efficiency = np.mean(np.abs(test_pred_interval[:,-1]-test_pred_interval[:,0]))

		return coverage, efficiency

	def coverage_distribution(self, y_test, test_pred_interval, plot_path, target_cov):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		coverages = np.zeros(n_points)
		for i in range(n_points):
			c = 0
			for j in range(self.test_hist_size):
				if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
					c += 1
			coverages[i] = c/self.test_hist_size

		avg_cov = np.mean(coverages)

		fig = plt.figure()
		plt.hist(coverages, bins = 50, stacked=False, density=True, color='lightsteelblue')
		plt.axvline(x=target_cov, color='k', linestyle='dashed', label=r'$1-\alpha$')
		plt.axvline(x=avg_cov, color='steelblue', linestyle='dashed', label='mean')
		
		plt.xlabel('coverage')
		plt.title(self.type_local+' loc cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		if self.type_local == 'gauss':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_marginal_coverage_distribution_eps={self.eps}.png")
		elif self.type_local == 'knn':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_marginal_coverage_distribution_k={self.knn}.png")
		plt.close()
	
		return coverages

	def cc_coverage_distribution(self, y_test, test_pred_interval, plot_path, target_cov):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		pos_coverages = []#np.zeros(n_points)
		neg_coverages = []#np.zeros(n_points)
		
		for i in range(n_points):
			c_pos, c_neg = 0, 0
			tot_pos, tot_neg = 0, 0
			for j in range(self.test_hist_size):
				if y_test_hist[i,j] >= 0:
					tot_pos += 1
				else:
					tot_neg += 1
				if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
					if y_test_hist[i,j] >= 0:
						c_pos += 1
					else:
						c_neg += 1
			if tot_pos > 0:
				pos_coverages.append(c_pos/tot_pos)
			if tot_neg > 0:
				neg_coverages.append(c_neg/tot_neg)

		pos_coverages = np.array(pos_coverages)
		neg_coverages = np.array(neg_coverages)
		
		avg_pos_cov = np.mean(pos_coverages)
		avg_neg_cov = np.mean(neg_coverages)
		fig = plt.figure()
		plt.hist(pos_coverages, bins = 50, stacked=False, density=True, color='cornflowerblue', label='pos')
		plt.hist(neg_coverages, bins = 50, stacked=False, density=True, color='lightcoral', label='neg')
		
		plt.axvline(x=target_cov, color='k', linestyle='dashed', label=r'$1-\alpha$')
		plt.axvline(x=avg_pos_cov, color='mediumblue', linestyle='dashed', label='pos mean')
		plt.axvline(x=avg_neg_cov, color='firebrick', linestyle='dashed', label='neg mean')
		
		plt.xlabel('cc coverage')
		plt.title(self.type_local+' loc cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		if self.type_local == 'gauss':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_cc_coverage_distribution_eps={self.eps}_fix.png")
		elif self.type_local == 'knn':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_cc_coverage_distribution_k={self.knn}_fix.png")
		else:
			print('warning: unrecognized localizer type!')
		plt.close()

	def get_coverage_efficiency_coupled(self, y_test, test_pred_interval1, test_pred_interval2):
		'''
		Compute the empirical coverage and the efficiency of the union of two prediction intervals (test_pred_interval1 and test_pred_interval2).
		y_test are the observed values of robustness
		'''
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		c = 0
		for i in range(n_points):
			for j in range(self.test_hist_size):
				# if it lies in at least one of the two intervals, i.e. in the union
				if (y_test_hist[i,j] >= test_pred_interval1[i,0] and y_test_hist[i, j] <= test_pred_interval1[i,-1]) or (y_test_hist[i,j] >= test_pred_interval2[i,0] and y_test_hist[i, j] <= test_pred_interval2[i,-1]):
					c += 1
		coverage = c/(n_points*self.test_hist_size)

		efficiency = np.mean(self.measure_efficiency_union(test_pred_interval1,test_pred_interval2))

		return coverage, efficiency

	def measure_efficiency_union(self, I1, I2):
		'''
		Measures the width of the interval resulting from the union of two intervals (I1 and I2)
		'''
		L1 = I1[:,1]-I1[:,0]
		L2 = I2[:,1]-I2[:,0]
		
		union_len = []
		for i in range(len(I1)):
			if I1[i,0] < I2[i,0]: 
				len_intersection = I1[i,1]-I2[i,0]
				if len_intersection > 0:
					union_len.append(L1[i]+L2[i]-len_intersection)
				else:	 
					union_len.append(L1[i]+L2[i])
			else:
				len_intersection = I2[i,1]-I1[i,0]
				if len_intersection > 0:
					union_len.append(L1[i]+L2[i]-len_intersection)
				else:	 
					union_len.append(L1[i]+L2[i])
		return np.array(union_len)

	def compute_accuracy_and_uncertainty(self, test_pred_interval, L_test):
		'''
		Computes the number of correct, uncertain and wrong prediction intervals and the number of false positives.
		L_test is the sign of the observed quantile interval (-1: negative, 0: uncertain, +1: positive)
		'''
		n_points = len(L_test)

		correct = 0
		wrong = 0
		uncertain = 0
		fp = 0

		for i in range(n_points):
			
			if L_test[i,2]: # sign +1
				if test_pred_interval[i,0] >= 0 and test_pred_interval[i,-1] > 0:
					correct += 1
				elif test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					uncertain += 1
				else:
					wrong +=1
			elif L_test[i,1]: # sign 0
				if test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					correct += 1
				else:
					wrong +=1
					if test_pred_interval[i,0] > 0:
						fp+= 1
			else: # sign -1
				if test_pred_interval[i,-1] <= 0 and test_pred_interval[i,0] < 0:
					correct += 1
				elif test_pred_interval[i,-1] >= 0 and test_pred_interval[i,0] <= 0:
					uncertain += 1
				else:
					wrong +=1
					fp += 1

		return correct/n_points, uncertain/n_points, wrong/n_points, fp/n_points


	def plot_loc_errorbars(self, y, qr_interval, cqr_interval, loc_cqr_interval, title_string, plot_path, extra_info = ''):
		'''
		Create barplots
		'''
		n_points_to_plot = 40
		
		
		n_test_points = len(y)//self.test_hist_size
		y_resh = np.reshape(y,(n_test_points,self.test_hist_size))
		y_resh = y_resh[:n_points_to_plot]
		yq = []
		yq_out = []
		xline_rep = []
		xline_rep_out = []
		for i in range(n_points_to_plot):

			lower_yi = np.quantile(y_resh[i], self.epsilon/2)
			upper_yi = np.quantile(y_resh[i], 1-self.epsilon/2)
			for j in range(self.test_hist_size):
				if y_resh[i,j] <= upper_yi and y_resh[i,j] >= lower_yi:
					yq.append(y_resh[i,j])
					xline_rep.append(i)
				else:
					yq_out.append(y_resh[i,j])
					xline_rep_out.append(i)					

		n_quant = qr_interval.shape[1]

		leg_lab = {'gauss':'Gauss', 'knn':'kNN'}
		xline = np.arange(n_points_to_plot)
		xline1 = np.arange(n_points_to_plot)+0.15
		xline2 = np.arange(n_points_to_plot)+0.3
		xline3 = np.arange(n_points_to_plot)+0.45
				
		fig = plt.figure(figsize=(20,4))
		plt.scatter(xline_rep_out, yq_out, c='peachpuff', s=6, alpha = 0.25)
		plt.scatter(xline_rep, yq, c='orange', s=6, alpha = 0.25,label='test')
		
		plt.plot(xline, np.zeros(n_points_to_plot), '-.', color='k')
		
		qr_med = qr_interval[:n_points_to_plot,1]
		qr_dminus = qr_med-qr_interval[:n_points_to_plot,0]
		qr_dplus = qr_interval[:n_points_to_plot,-1]-qr_med
		plt.errorbar(x=xline1, y=qr_med, yerr=[qr_dminus,qr_dplus],  color = 'c', fmt='o', capsize = 4, label='QR')
		

		cqr_med = (cqr_interval[:n_points_to_plot,0]+cqr_interval[:n_points_to_plot,-1])/2
		cqr_dminus = cqr_med-cqr_interval[:n_points_to_plot,0]
		cqr_dplus = cqr_interval[:n_points_to_plot,-1]-cqr_med
		plt.errorbar(x=xline2, y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'blue', fmt='none', capsize = 4,label='CQR')

		loc_cqr_med = (loc_cqr_interval[:n_points_to_plot,0]+loc_cqr_interval[:n_points_to_plot,-1])/2
		loc_cqr_dminus = loc_cqr_med-loc_cqr_interval[:n_points_to_plot,0]
		loc_cqr_dplus = loc_cqr_interval[:n_points_to_plot,-1]-loc_cqr_med
		plt.errorbar(x=xline3, y=loc_cqr_med, yerr=[loc_cqr_dminus,loc_cqr_dplus],  color = 'darkviolet', fmt='none', capsize = 4,label=leg_lab[self.type_local]+'-CQR')

		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		if self.type_local == 'gauss':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_"+extra_info+f"_errorbar_eps={self.eps}.png")
		elif self.type_local == 'knn':
			fig.savefig(plot_path+f"/reloc_{self.type_local}_"+extra_info+f"_errorbar_k={self.knn}.png")
		else:
			print('warning: unrecognized localizer type!')
		plt.close()


	def plot_comb_errorbars(self, y, cqr_interval, title_string, plot_path, extra_info = ''):
		'''
		Create barplots for conjuction of properties:
		when combining two prediction intervals there is no qr_interval and no median
		'''

		n_points_to_plot = 40
		
		n_test_points = len(y)//self.test_hist_size
		y_resh = np.reshape(y,(n_test_points,self.test_hist_size))
		y_resh = y_resh[:n_points_to_plot]
		yq = []
		yq_out = []
		xline_rep = []
		xline_rep_out = []
		for i in range(n_points_to_plot):

			lower_yi = np.quantile(y_resh[i], self.epsilon/2)
			upper_yi = np.quantile(y_resh[i], 1-self.epsilon/2)
			for j in range(self.test_hist_size):
				if y_resh[i,j] <= upper_yi and y_resh[i,j] >= lower_yi:
					yq.append(y_resh[i,j])
					xline_rep.append(i)
				else:
					yq_out.append(y_resh[i,j])
					xline_rep_out.append(i)					



		xline = np.arange(n_points_to_plot)
		xline1 = np.arange(n_points_to_plot)+0.2
		xline2 = np.arange(n_points_to_plot)+0.4
				
		fig = plt.figure(figsize=(20,4))
		plt.scatter(xline_rep_out, yq_out, c='peachpuff', s=6, alpha = 0.25)
		plt.scatter(xline_rep, yq, c='orange', s=6, alpha = 0.25,label='test')
		
		plt.plot(xline, np.zeros(n_points_to_plot), '-.', color='k')

		cqr_med = (cqr_interval[:n_points_to_plot,-1]+cqr_interval[:n_points_to_plot,0])/2
		cqr_dminus = cqr_med-cqr_interval[:n_points_to_plot,0]
		cqr_dplus = cqr_interval[:n_points_to_plot,-1]-cqr_med
		plt.errorbar(x=xline2, y=cqr_med, yerr=[cqr_dminus,cqr_dplus], fmt = 'none', capsize = 4, color = 'darkviolet', label='Conj of CQRs')
		
		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+"/"+extra_info+"_reloc_errorbar.png")
		plt.close()
