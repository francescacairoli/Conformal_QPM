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

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class CC_CQR():
	'''
	The Class conditional CQR class implements Conformalized Quantile Regression, i.e. it applies CP to a QR
	Inputs: 
		- Xc, Yc: the calibration set
		- trained_qr_model: pre-trained quantile regressor
		- quantiles: the quantiles used to train the quantile regressor
		- test_hist_size, cal_hist_size: number of observations per point in the test and calibration set respectively
		- comb_flag = False: performs normal CQR over a single property 
		- comb_flag = True: combine the prediction intervals of the CQR of two properties
	'''
	def __init__(self, Xc, Yc, trained_qr_model, test_hist_size = 2000, cal_hist_size = 50, quantiles = [0.05, 0.95], comb_flag = False):
		super(CC_CQR, self).__init__()

		self.Xc = Xc 
		self.Yc = Yc
		self.trained_qr_model = trained_qr_model
		
		self.q = len(Yc) # number of points in the calibration set
		self.test_hist_size = test_hist_size
		self.cal_hist_size = cal_hist_size
		self.quantiles = quantiles
		self.epsilon = 2*quantiles[0]
		self.M = len(quantiles) # number of quantiles
		self.col_list = ['yellow', 'orange', 'red', 'orange', 'yellow']
		self.comb_flag = comb_flag

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
		n = pred_interval.shape[0] # number of states
		m = len(y)  # m = n x self.cal_hist_size
		ncm = np.empty(m+1)

		c = 0		
		for i in range(n):
			for j in range(self.cal_hist_size):
			
				ncm[c] = max(pred_interval[i,0]-y[c], y[c]-pred_interval[i,-1]) # pred_interval[i,0] = q_lo(x), pred_interval[i,1] = q_hi(x)
				c += 1	
		ncm[-1] = math.inf

		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm

	def get_calibr_cc_nonconformity_scores(self, y, pi, sorting = True):
		'''
		Compute the nonconformity scores over the calibration set
		if sorting = True the returned scores are ordered in a descending order
		'''
		n = pi.shape[0] # number of states
		m = len(y)  # m = n x self.cal_hist_size
		ncm_plus = []
		ncm_minus = []

		
		c = 0		
		for i in range(n):
			for j in range(self.cal_hist_size):
			
				if y[c] >= 0:
					ncm_plus.append(max(pi[i,0]-y[c], y[c]-pi[i,-1]))
				else:
					ncm_minus.append(max(pi[i,0]-y[c], y[c]-pi[i,-1]))
				c += 1
		ncm_plus.append(math.inf)
		ncm_minus.append(math.inf)

		ncm_plus = np.array(ncm_plus)
		ncm_minus = np.array(ncm_minus)

		#print("-----", ncm_plus.shape, ncm_minus.shape)
		if sorting:
			ncm_plus = np.sort(ncm_plus)[::-1] # descending order
			ncm_minus = np.sort(ncm_minus)[::-1] 
		return ncm_plus, ncm_minus


	def get_scores_thresholds(self):
		'''
		This method extract the threshold value tau (the quantile at level epsilon) from the 
		calibration nonconformity scores (computed by the 'self.get_calibr_nonconformity_scores' method)
		'''
		self.calibr_pred = self.get_pred_interval(self.Xc)

		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores(self.Yc, self.calibr_pred)

		self.calibr_scores_plus, self.calibr_scores_minus = self.get_calibr_cc_nonconformity_scores(self.Yc, self.calibr_pred)
		
		print("Nb of calibr scores = ", len(self.calibr_scores))

		print("Nb of POSITIVE calibr scores = ", len(self.calibr_scores_plus))
		print("Nb of NEGATIVE calibr scores = ", len(self.calibr_scores_minus))

		Q = (1-self.epsilon)*(1+1/len(self.calibr_scores))
		if len(self.calibr_scores_plus) > 0:
			Qp = min((1-self.epsilon)*(1+1/len(self.calibr_scores_plus)),1)
			self.tau_plus = np.quantile(self.calibr_scores_plus, Qp)
		
		else:
			self.tau_plus = math.inf
		if len(self.calibr_scores_minus) > 0:
			Qm = min((1-self.epsilon)*(1+1/len(self.calibr_scores_minus)),1)
			self.tau_minus = np.quantile(self.calibr_scores_minus, Qm)

		else:
			self.tau_minus = math.inf

		self.tau = np.quantile(self.calibr_scores, Q)
		
		print("self.tau: ", self.tau)
		print("self.tau_plus: ", self.tau_plus)
		print("self.tau_minus: ", self.tau_minus)


	def get_cpi(self, inputs, pi_flag = False):
		'''
		Returns the conformalized prediction interval (cpi) by enlarging the 
		QR prediction interval by adding an subtracting tau from the lower and upper bound resp.
		'''
		pi = self.get_pred_interval(inputs)
		self.get_scores_thresholds()

		n = pi.shape[0]
		n_quant = pi.shape[1]


		cpi = np.vstack((pi[:,0],pi[:,-1])).T
		cpi[:,0] -= self.tau
		cpi[:,-1] += self.tau

		pos_cpi = np.vstack((pi[:,0],pi[:,-1])).T
		pos_cpi[:,0] -= self.tau_plus
		pos_cpi[:,-1] += self.tau_plus

		neg_cpi = np.vstack((pi[:,0],pi[:,-1])).T
		neg_cpi[:,0] -= self.tau_minus
		neg_cpi[:,-1] += self.tau_minus


		cc_cpi = np.empty((n,2))
		
		for i in range(n):
			if pos_cpi[i,0] >= 0 and neg_cpi[i,0] >= 0:
				cc_cpi[i] = pos_cpi[i]
				
			elif neg_cpi[i,-1] <= 0 and pos_cpi[i,-1] < 0:
				cc_cpi[i] = neg_cpi[i]
				
			else: # across zero
				cc_cpi[i,0] = neg_cpi[i,0]
				cc_cpi[i,-1] = pos_cpi[i,-1]


		if pi_flag:
			return cpi, cc_cpi, pi
		else:
			return cpi, cc_cpi

	def get_cc_cpi(self, inputs, pi_flag = False):
		'''
		Returns the conformalized prediction interval (cpi) by enlarging the 
		QR prediction interval by adding an subtracting tau from the lower and upper bound resp.
		'''
		pi = self.get_pred_interval(inputs)
		self.get_scores_thresholds()


		n = pi.shape[0]

		cpi = np.vstack((pi[:,0],pi[:,-1])).T
		cpi[:,0] -= self.tau
		cpi[:,-1] += self.tau

		pos_cpi = np.vstack((pi[:,0],pi[:,-1])).T
		pos_cpi[:,0] -= self.tau_plus
		pos_cpi[:,-1] += self.tau_plus

		neg_cpi = np.vstack((pi[:,0],pi[:,-1])).T
		neg_cpi[:,0] -= self.tau_minus
		neg_cpi[:,-1] += self.tau_minus


		if pi_flag:
			return cpi, pos_cpi, neg_cpi, pi
		else:
			return cpi, pos_cpi, neg_cpi
		



	def coverage_distribution(self, y_test, test_pred_interval, plot_path, target_cov):
		'''
		Compute the empirical coverage and the efficiency of a prediction interval (test_pred_interval).
		y_test are the observed target values
		'''
		nbins = 100
		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		coverages = np.zeros(n_points)
		for i in range(n_points):
			c = 0
			for j in range(self.test_hist_size):
				if (y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]):
					c += 1
			coverages[i] = c/self.test_hist_size

		avg_cov = np.mean(coverages)
		
		fig = plt.figure()
		plt.hist(coverages, bins = nbins, stacked=False, density=False, color='lightsteelblue',range=[0,1])
		plt.axvline(x=target_cov,color='k', linestyle='dashed', label=r'$1-\alpha$')
		plt.axvline(x=avg_cov,color='steelblue', linestyle='dashed', label='mean')
		
		plt.xlabel('coverage')
		plt.title('cb cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		fig.savefig(plot_path+"/CB_marginal_coverage_distribution.png")
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
		nbins = 100
		for i in range(n_points):
			c_pos, c_neg = 0, 0
			tot_pos, tot_neg = 0, 0
			for j in range(self.test_hist_size):
				if y_test_hist[i,j] >= 0:
					tot_pos += 1
					if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
						c_pos += 1
				else:
					tot_neg += 1
					if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
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
		plt.hist(pos_coverages, bins = nbins, stacked=False, density=False, color='cornflowerblue', label='pos',range=[0,1])
		plt.hist(neg_coverages, bins = nbins, stacked=False, density=False, color='lightcoral', label='neg',range=[0,1])
		
		plt.axvline(x=target_cov,color='k', linestyle='dashed', label=r'$1-\alpha$')
		plt.axvline(x=avg_pos_cov,color='mediumblue', linestyle='dashed', label='pos mean')
		plt.axvline(x=avg_neg_cov, color='firebrick', linestyle='dashed', label='neg mean')
		
		plt.xlabel('cc coverage')
		plt.title('cb cpi')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		fig.savefig(plot_path+"/CB_cc_coverage_distribution_fix.png")
		plt.close()



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


	def plot_errorbars(self, y, qr_interval, cqr_interval, cc_cqr_interval, title_string, plot_path, extra_info = ''):
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
		
		cqr_med = (cqr_interval[:n_points_to_plot,0]+cqr_interval[:n_points_to_plot,-1])/2
		cqr_dminus = cqr_med-cqr_interval[:n_points_to_plot,0]
		cqr_dplus = cqr_interval[:n_points_to_plot,-1]-cqr_med
		plt.errorbar(x=xline2, y=cqr_med, yerr=[cqr_dminus,cqr_dplus],  color = 'blue', fmt='none', capsize = 4,label='CQR')

		cc_cqr_med = (cc_cqr_interval[:n_points_to_plot,0]+cc_cqr_interval[:n_points_to_plot,-1])/2
		cc_cqr_dminus = cc_cqr_med-cc_cqr_interval[:n_points_to_plot,0]
		cc_cqr_dplus = cc_cqr_interval[:n_points_to_plot,-1]-cc_cqr_med
		
		plt.errorbar(x=xline3, y=cc_cqr_med, yerr=[cc_cqr_dminus,cc_cqr_dplus],  color = 'darkviolet', fmt='none', capsize = 4,label='CB-CQR')
		

		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+"/"+extra_info+"_cb_union_errorbar.png")
		plt.close()

	def plot_cc_errorbars(self, y, qr_interval, cqr_interval, pos_cqr_interval, neg_cqr_interval, title_string, plot_path, extra_info = ''):
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
		xline4 = np.arange(n_points_to_plot)+0.5
				
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

		pos_cqr_med = (pos_cqr_interval[:n_points_to_plot,0]+pos_cqr_interval[:n_points_to_plot,-1])/2
		pos_cqr_dminus = pos_cqr_med-pos_cqr_interval[:n_points_to_plot,0]
		pos_cqr_dplus = pos_cqr_interval[:n_points_to_plot,-1]-pos_cqr_med
		
		plt.errorbar(x=xline3, y=pos_cqr_med, yerr=[pos_cqr_dminus,pos_cqr_dplus],  color = 'cornflowerblue', fmt='none', capsize = 4,label='Pos-CQR')
		
		neg_cqr_med = (neg_cqr_interval[:n_points_to_plot,0]+neg_cqr_interval[:n_points_to_plot,-1])/2
		neg_cqr_dminus = neg_cqr_med-neg_cqr_interval[:n_points_to_plot,0]
		neg_cqr_dplus = neg_cqr_interval[:n_points_to_plot,-1]-neg_cqr_med
		
		plt.errorbar(x=xline4, y=neg_cqr_med, yerr=[neg_cqr_dminus,neg_cqr_dplus],  color = 'lightcoral', fmt='none', capsize = 4,label='Neg-CQR')
		
		plt.ylabel('robustness')
		plt.title(title_string)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		fig.savefig(plot_path+"/"+extra_info+"_cb_errorbar.png")
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
		fig.savefig(plot_path+"/"+extra_info+"_errorbar.png")
		plt.close()
