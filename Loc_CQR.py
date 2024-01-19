import copy
import torch
import numpy as np
import scipy.special
import scipy.spatial
from numpy.random import rand
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from torch.autograd import Variable

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
	def __init__(self, Xc, Yc, trained_qr_model, eps= 0.1, test_hist_size = 200, cal_hist_size = 50, quantiles = [0.05, 0.95], comb_flag = False,plots_path=''):
		super(Loc_CQR, self).__init__()

		self.Yc = Yc
		self.Xc = Xc
		self.eps = eps
		self.trained_qr_model = trained_qr_model
		self.q = len(Yc) # number of points in the calibration set
		self.test_hist_size = test_hist_size
		self.cal_hist_size = cal_hist_size
		self.quantiles = quantiles
		self.epsilon = 2*quantiles[0]
		self.Q = (1-self.epsilon)*(1+1/self.q)
		
		self.M = len(quantiles) # number of quantiles
		self.col_list = ['yellow', 'orange', 'red', 'orange', 'yellow']
		self.comb_flag = comb_flag # default: False
		self.plots_path = plots_path

	def get_pred_interval(self, inputs):
		'''
		Apply the trained QR to inputs and returns the QR prediction interval
		'''
		
		quantiles = self.trained_qr_model(Variable(FloatTensor(inputs))).cpu().detach().numpy()
		return quantiles
		


	def get_calibr_nonconformity_scores(self, y, pred_interval, sorting = True):
		'''
		Compute the nonconformity scores over the calibration set
		if sorting = True the returned scores are ordered in a descending order
		'''
		m = len(y)
		n = pred_interval.shape[0] # number of states
		ncm = np.empty(m)
		
		c = 0
		for i in range(n):
			for j in range(self.cal_hist_size):
			
				ncm[c] = max(pred_interval[i,0]-y[c], y[c]-pred_interval[i,-1]) # pred_interval[i,0] = q_lo(x), pred_interval[i,1] = q_hi(x)
				c += 1
		ncm_sort = np.sort(ncm)[::-1] # descending order
		self.tau = np.quantile(ncm_sort, self.Q)

		if False:
			fig = plt.figure()
			plt.scatter(np.arange(len(ncm_)), ncm, color='g')
			plt.scatter(self.epsilon*(1+1/self.q)*len(ncm), self.tau,color='r')
			plt.title('calibr ncm')
			plt.tight_layout()
			plt.savefig(self.plots_path+'/calibr_nonconf_scores.png')
			plt.close()

			
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
			

		return ncm


	def get_calibr_scores(self):

		self.calibr_pred = self.get_pred_interval(self.Xc)
		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores(self.Yc, self.calibr_pred, sorting=False)
	
	def compute_likelihood(self, x_star, xi):

		
		k = len(x_star)
		inv_eps = 1/self.eps
		inv_sigma = inv_eps*np.eye(k)
		
		vol = 1#np.abs((-2 +1.5)*(-4+3.5)*(0.5-1))
		den = np.sqrt((2*np.pi*self.eps)**k)

		const = vol/den
		diff = xi-x_star
		
		ratio = const*np.exp(-np.dot(diff.reshape(1,k),np.dot(inv_sigma,diff))/2)
		
		return ratio[0]

	def weight_calibr_scores(self, x_star):
		n = len(self.Xc)

		wncs = np.empty(self.q)

		c = 0
		for i in range(n):
			r_i = self.compute_likelihood(x_star, self.Xc[i])
			for j in range(self.cal_hist_size):
				wncs[c] = self.calibr_scores[c]*r_i
				c += 1

		return np.sort(wncs)[::-1] # descending order



	def get_scores_threshold(self,x_star):
		'''
		This method extract the threshold value tau (the quantile at level epsilon) from the 
		calibration nonconformity scores (computed by the 'self.get_calibr_nonconformity_scores' method)
		'''
		weight_cal_scores = self.weight_calibr_scores(x_star)
		tau_star = np.quantile(weight_cal_scores, self.Q)
		
		if False:
			fig = plt.figure()
			plt.scatter(np.arange(len(weight_cal_scores)), weight_cal_scores, color='g')
			plt.scatter(self.epsilon*(1+1/self.q)*len(weight_cal_scores), tau_star,color='r')
			plt.title('weight calibr ncm')
			plt.tight_layout()
			plt.savefig(self.plots_path+'/weight_calibr_nonconf_scores.png')
			plt.close()

		return tau_star


	def get_cpi(self, inputs, pi_flag = False):
		'''
		Returns the conformalized prediction interval (cpi) by enlarging the 
		QR prediction interval by adding an subtracting tau from the lower and upper bound resp.
		'''
		pi = self.get_pred_interval(inputs)
		n_quant = pi.shape[1]

		cpi = pi[:,0]-self.tau
		for jj in range(1,n_quant-1):
			cpi = np.vstack((cpi, pi[:,jj]))
			
		cpi = np.vstack((cpi, pi[:,-1]+self.tau))

		loc_cpi = np.empty(pi.shape)

		for j in range(pi.shape[0]): 
			
			tau_j = self.get_scores_threshold(inputs[j])

			loc_cpi[j,0] = pi[j,0]-tau_j 
			for k in range(1,n_quant-1):
				loc_cpi[j,k] = pi[j,k]
			loc_cpi[j,-1] = pi[j,-1]+tau_j
		
		if pi_flag:
			return cpi.T, loc_cpi, pi
		else:
			return cpi.T, loc_cpi

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
		ym = 15
		fig = plt.figure()
		plt.hist(coverages, bins = 50, stacked=False, density=True, color='lightsteelblue')
		plt.vlines(x=target_cov, ymin=0,ymax=ym,colors='k', linestyles='dashed', label=r'$1-\alpha$')
		plt.vlines(x=avg_cov, ymin=0,ymax=ym,colors='steelblue', linestyles='dashed', label='mean')
		
		plt.xlabel('coverage')
		plt.title('loc cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		fig.savefig(plot_path+"/loc_marginal_coverage_distribution.png")
		plt.close()
	
		return coverages

	def cc_coverage_distribution(self, y_test, test_pred_interval, plot_path, target_cov):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		pos_coverages = np.zeros(n_points)
		neg_coverages = np.zeros(n_points)
		
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
			if tot_pos == 0:
				pos_coverages[i] = 1
			else:
				pos_coverages[i] = c_pos/tot_pos
			if tot_neg == 0:
				neg_coverages[i] = 1
			else:
				neg_coverages[i] = c_neg/tot_neg

		ym = 15
		avg_pos_cov = np.mean(pos_coverages)
		avg_neg_cov = np.mean(neg_coverages)
		fig = plt.figure()
		plt.hist(pos_coverages, bins = 50, stacked=False, density=True, color='cornflowerblue', label='pos')
		plt.hist(neg_coverages, bins = 50, stacked=False, density=True, color='lightcoral', label='neg')
		
		plt.vlines(x=target_cov, ymin=0,ymax=ym,colors='k', linestyles='dashed', label=r'$1-\alpha$')
		plt.vlines(x=avg_pos_cov, ymin=0,ymax=ym,colors='mediumblue', linestyles='dashed', label='pos mean')
		plt.vlines(x=avg_neg_cov, ymin=0,ymax=ym,colors='firebrick', linestyles='dashed', label='neg mean')
		
		plt.xlabel('cc coverage')
		plt.title('loc cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		fig.savefig(plot_path+"/loc_cc_coverage_distribution.png")
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
		

		cqr_dminus = qr_med-cqr_interval[:n_points_to_plot,0]
		cqr_dplus = cqr_interval[:n_points_to_plot,-1]-qr_med
		plt.errorbar(x=xline2, y=qr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'blue', fmt='o', capsize = 4,label='CQR')

		loc_cqr_dminus = qr_med-loc_cqr_interval[:n_points_to_plot,0]
		loc_cqr_dplus = loc_cqr_interval[:n_points_to_plot,-1]-qr_med
		plt.errorbar(x=xline3, y=qr_med, yerr=[loc_cqr_dminus,loc_cqr_dplus],  color = 'darkviolet', fmt='o', capsize = 4,label='Loc-CQR')

		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+"/"+extra_info+f"_errorbar_merged_eps={self.eps}.png")
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
		fig.savefig(plot_path+"/"+extra_info+"_errorbar_merged.png")
		plt.close()
