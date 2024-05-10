import numpy as np
import Criticality as cr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import holoviews as hv
hv.extension('bokeh')
from matplotlib import rc
from matplotlib import rc,rcParams
# rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r"\usepackage{sfmath} \boldmath"


# plt.rcParams["font.family"] = "Serif"

def AV_analysis_ExponentErrorComments(burst, T, bm, tm, flag = 1, EX_burst = 1, EX_time = 1):
# This function calculate exponents for PDF(AVsize), PDF(AVdura), and
# scaling relation. When flag == 1 (default),
# Tranlsated by Lizzie Tilden 5/29/19
# Edited by Yifan Xu 11/18/19

	Result = {}

	# calculate exponent and other parameters for AVsize.
	if EX_burst:       
		try:
# 			print('Trying to exclude outliers from size data....')
			burstMax, burstMin, alpha = cr.EXCLUDE(burst[burst < np.power(np.max(burst),0.8)],setmin=bm)# get burstMin and burstMax    
			print('alpha for size=',alpha)                       
			idx_burst = np.where(np.logical_and(burst >= burstMin,burst <= burstMax+1))[0]
		except ValueError:
			print('could not minimize KS for burst size...wants to exclude all data.')
			burstMax = np.power(np.max(burst),0.8)
			burstMin = bm
			idx_burst = np.where(np.logical_and(burst >= burstMin,burst < burstMax))[0]
	else:
		burstMax = np.max(burst)
		burstMin = bm
		idx_burst = np.where(np.logical_and(burst >= burstMin,burst <= burstMax))[0]
		
	alpha, xmin, xmax, L = cr.tplfit(burst[idx_burst], burstMin) 
	print('alpha for size=',alpha)
	if burstMin > xmin:
		print('burstMin is larger than xmin in tplfit for burst')

	Result['burst'] = burst
	Result['alpha'] = alpha
	Result['xmin'] = burstMin
	Result['xmax'] = burstMax
	
	if flag == 2:
	    ##### calculate pvalue for null hypothesis test #####
	    Result['P_burst'], ks, hax_burst = cr.pvaluenew(burst[idx_burst],40)
	    hax_burst.axes[0].set_xlabel('Size (S)', fontsize = 16)
	    hax_burst.axes[0].set_ylabel('Prob(size < S)', fontsize = 16)

	
	elif flag == 3:
		#plot distribution together with fitted distribution
    		############### Plot PDF ####################
		print('plotting burst PDF')
		fig1, ax1 = plt.subplots(nrows = 1, ncols = 3, figsize = [9, 5])
		pdf = np.histogram(burst, bins = np.arange(0, np.max(burst)+2))[0]
		ax1[0].plot(pdf/np.sum(pdf), marker = 'o', markersize = 4, linestyle = 'None', color = '#2138ab', alpha = 0.75)
		ax1[0].set_yscale('log')
		ax1[0].set_xscale('log')
 

		############### Plot fitted PDF #################
		
		x = np.arange(burstMin, burstMax+1)
		y = (np.size(np.where(burst == xmin+6)[0])/np.power(xmin+6, -alpha))*np.power(x, -alpha)
		y = y/np.sum(pdf)

		ax1[0].plot(x,y, color = '#c5c9c7');
		ax1[0].set_xlabel('S', fontsize = 16,fontweight='bold')
		ax1[0].set_ylabel('PDF(S)', fontsize = 16,fontweight='bold')
		ax1[0].set_title('\u03C4 = ' + str(np.round(alpha[0], 3)), fontsize = 16,fontweight='bold')
		ax1[0].tick_params(axis='x', labelsize=14)
		ax1[0].tick_params(axis='y', labelsize=14)
		plt.ion()

	if EX_time:
		#calculate exponent and other parameters for AVdura.
		try:
# 			print('Trying to exclude outliers from duration data....')
			tMax, tMin, beta = cr.EXCLUDE(T[T < np.power(np.max(T),0.8)],setmin = tm)
			print('tau for duration=',beta)
			if tMax < 40:
				tMax = np.floor(np.power(np.max(T),0.9))
# 				print("tMax < 40, use new tMax")
# 				print(tMax)

			idx_time = np.where(np.logical_and(T >= tMin,T <= tMax + 1))[0]
		except ValueError:
			print('could not minimize KS for burst size...wants to exclude all data.')
			tMax = np.power(np.max(T),0.8)
			tMin = tm
			idx_time = np.where(np.logical_and(T >= tMin,T < tMax))[0]
	else:
		tMax = np.max(T)
		tMin = tm
		idx_time = np.where(np.logical_and(T >= tMin,T <= tMax))[0]
	
	beta, tmin, tmax, L = cr.tplfit(T[idx_time], tMin);
	print('tau for duration=',beta)
	if tMin > tmin:
		print('TMin is larger than tmin in tplfit for burst')

	Result['T'] = T
	Result['beta'] = beta
	Result['tmin'] = tMin
	Result['tmax'] = tMax

	if flag == 2:
    	##### calculate pvalue for null hypothesis test #####
		Result['P_t'], ks, hax_time  = cr.pvaluenew(T[idx_time],30)
		hax_time.axes[0].set_xlabel('Duration (D)', fontsize = 16)
		hax_time.axes[0].set_ylabel('Prob(size < D)', fontsize = 16)
		#hax_time.suptitle('Avalanche Duration')

	elif flag == 3:
    	# plot distribution together with fitted distribution
    	############### Plot PDF ####################
		print('plotting time TDF')
		tdf = np.histogram(T,bins = np.arange(0, np.max(T)+2))[0]
		ax1[1].plot(tdf/np.sum(tdf), marker = 'o', markersize = 4, linestyle = 'None', color = '#48D1CC', alpha = 0.75)
		ax1[1].set_yscale('log')
		ax1[1].set_xscale('log')
		sns.despine()

		 ############### Plot fitted PDF #################
		x = np.arange(tMin,tMax+1);
		y = np.size(np.where(T == tmin+4))/(np.power(tmin+4,-beta))*np.power(x,-beta)
		y = y/np.sum(tdf)
		ax1[1].plot(x,y, color = '#c5c9c7')
		ax1[1].set_xlabel('D', fontsize = 16,fontweight='bold')
		ax1[1].set_ylabel('PDF(D)', fontsize = 16,fontweight='bold')
		ax1[1].set_title('\u03B1 = ' + str(np.round(beta[0], 3)), fontsize = 16,fontweight='bold')
		ax1[1].tick_params(axis='x', labelsize=14)
		ax1[1].tick_params(axis='y', labelsize=14)

        ############ New plots with holoviews ############
# # 		fig2, ax2 = plt.subplots(nrows = 1, ncols = 3, figsize = [10, 6])
# 		scatter1 = hv.Scatter((x, y))
# 		curve1 = hv.Curve(tdf/np.sum(tdf))
# #         curve2 = hv.Curve(scatter2).opts(line_dash=(4, 4), color='orange')
# #         curve3 = hv.Curve(scatter3).opts(color='orange')       
# #         example1 = scatter1 * scatter2 * scatter3
# 		example2 = scatter1 *  curve1 
# 		example2.relabel("Another Legend Example")
# 		example2.opts(logx=True, logy=True, shared_axes=False)



 

	################## scaling relation #####################
	TT = np.arange(1, np.max(T)+1)
	Sm = []
	########### Calculate average size for each duration #########
	for i in np.arange(0,np.size(TT)):
		Sm.append(np.mean(burst[np.where(T==TT[i])[0]]))
	Sm = np.asarray(Sm)
	 ################## get fit and pre exponent ##################
	Loc=np.where(Sm>0)[0]
	TT=TT[Loc]
	Sm=Sm[Loc]
	
	fit_sigma = np.polyfit(np.log(TT[np.intersect1d(np.where(TT>tMin)[0], np.where(TT<tMin+60)[0])]), np.log(Sm[np.intersect1d(np.where(TT>tMin)[0], np.where(TT<tMin+60)[0])]), 1);
	
	sigma = (beta - 1)/(alpha - 1)
	print('beta is:',sigma)
	print('fitted beta is:',fit_sigma)


	Result['pre'] = sigma
	Result['fit'] = fit_sigma
	Result['df'] = np.abs(sigma - fit_sigma[0])
	print('DCC is:',Result['df'])
	# Result['TT'] = TT
	# Result['Sm'] = Sm

	if flag == 3:
	    ############# Plot scaling relation/fitted/predicted ###########
		print('plotting scaling relation')
		m = fit_sigma[0]
		c = fit_sigma[1]
		ax1[2].plot(TT, ((np.power(TT,sigma)/np.power(TT[7],sigma))*Sm[7]), label = '$\u03B2_{pred}$ ', color = '#4b006e')
		ax1[2].plot(TT, (np.power(TT,fit_sigma[0])/np.power(TT[7],fit_sigma[0])*Sm[7]),'b', label = '$\u03B2_{fit}$', linestyle = '--', color = '#826d8c')
		ax1[2].plot(TT, Sm,'o', color = '#fb7d07', alpha = 0.75)
		ax1[2].set_xscale('log')
		ax1[2].set_yscale('log')
		ax1[2].set_ylabel('<S>',fontsize = 16,fontweight='bold')
		ax1[2].set_xlabel('D',fontsize = 16,fontweight='bold')
		ax1[2].set_title('DCC = ' + str(np.round(Result['df'][0], 3)),fontsize = 16,fontweight='bold')
		# ax1[0].set_xlim([5,1000])
		# ax1[1].set_xlim([0, 200])
		# ax1[2].set_xlim([0, 150])
		plt.tight_layout()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.legend(fontsize=14)
		plt.show()

 

		return Result, fig1,np.round(Result['df'][0], 3),fit_sigma, alpha, beta
	elif flag == 2:
		return Result, hax_burst, hax_time,fit_sigma,alpha, beta
	else:
		return Result,fit_sigma,alpha, beta





