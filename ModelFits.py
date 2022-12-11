from numpy.polynomial import Polynomial
import numpy as np
from astropy.table import Table
import sys
from datetime import datetime
import matplotlib as plt
from matplotlib import rc, ticker, rcParams
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})

#%matplotlib inline
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
import matplotlib.pyplot as plt
plt.ion()
rcParams['axes.linewidth'] = 1.3
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['xtick.minor.visible'] = True
rcParams['xtick.minor.width'] = 1
rcParams['xtick.direction'] = 'inout'
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.width'] = 1
rcParams['ytick.direction'] = 'out'

calFile = sys.argv[1]

aboutBoolean = input('Is there anything you want to remember about this fit? If yes, write True \n')
message = ''
if aboutBoolean == 'True':
    message = input('write it here:')
aboutTxt = 'about' + dt_string + '.txt'
f = open(aboutTxt,"w")
f.write(calFile+' '+dt_string+' '+message+ '\n')
f.close()

stacked_shears = Table.read(calFile, format='fits')
stacked_shears.colnames
diff_gtan = stacked_shears['mean_gtan'] - stacked_shears['mean_nfw_gtan']
diff_gcross = stacked_shears['mean_gcross'] - stacked_shears['mean_nfw_gcross']
nfw_gtan = stacked_shears['mean_nfw_gtan']
nfw_gcross = stacked_shears['mean_nfw_gcross']

wl_regime = 0.075
wl = stacked_shears['mean_nfw_gtan'] < wl_regime
weight = np.sqrt(stacked_shears['err_gtan']**2 + stacked_shears['err_nfw_gtan']**2)
xweight = np.sqrt(stacked_shears['err_gcross']**2 + stacked_shears['err_nfw_gcross']**2)
### Time to do the fit!
### Start with g_tan 

xx = np.linspace(-1, 1, 50)

fit = Polynomial.fit(x=nfw_gtan, y=diff_gtan, deg=1, full=True)
y_fit = fit[0].coef[0] + xx*fit[0].coef[1]
f = open(aboutTxt, "a")
f.write(f'##\n##  PolyFit to all gtan values:\n##')
f.write(f'\t{fit[0]}')
f.write(f'\tsum of squared fit residuals: {fit[1][0][0]}\n')
f.close()

wfit = Polynomial.fit(x=nfw_gtan, y=diff_gtan, deg=1, full=True, w=1./weight)
y_wfit = wfit[0].coef[0] + xx*wfit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted PolyFit to all gtan values:\n##')
f.write(f'\t{wfit[0]}')
f.write(f'\tsum of squared fit residuals: {wfit[1][0][0]}\n')
f.close()


wl_fit = Polynomial.fit(x=nfw_gtan[wl], y=diff_gtan[wl], deg=1, full=True)
y_wl_fit = wl_fit[0].coef[0] + xx*wl_fit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  PolyFit to gtan < {wl_regime} :\n##')
f.write(f'\t{wl_fit[0]}')
f.write(f'\tsum of squared fit residuals: {wl_fit[1][0][0]}\n')
f.close()

wl_wfit = Polynomial.fit(x=nfw_gtan[wl], y=diff_gtan[wl], deg=1, full=True, w=1./weight[wl])
y_wl_wfit = wl_fit[0].coef[0] + xx*wl_fit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted PolyFit to gtan < {wl_regime} :\n##')
f.write(f'\t{wl_wfit[0]}')
f.write(f'\tsum of squared fit residuals: {wl_wfit[1][0][0]}\n')
f.close()
###
### Repeat with g_cross
### 

xx = np.linspace(-1, 1, 50)

xfit = Polynomial.fit(x=nfw_gtan, y=diff_gcross, deg=1, full=True)
y_xfit = xfit[0].coef[0] + xx*xfit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  PolyFit to gcross for all gtan values:\n##')
f.write(f'\t{fit[0]}')
f.write(f'\tsum of squared fit residuals: {xfit[1][0][0]}\n')
f.close()

wxfit = Polynomial.fit(x=nfw_gtan, y=diff_gcross, deg=1, full=True, w=1./xweight)
y_wxfit = wxfit[0].coef[0] + xx*wxfit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted PolyFit to grcoss for all gtan values:\n##')
f.write(f'\t{wfit[0]}')
f.write(f'\tsum of squared fit residuals: {wfit[1][0][0]}\n')
f.close()

wl_xfit = Polynomial.fit(x=nfw_gtan[wl], y=diff_gcross[wl], deg=1, full=True)
y_wl_xfit = wl_xfit[0].coef[0] + xx*wl_xfit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  PolyFit to gcross in gtan < {wl_regime} regime:\n##')
f.write(f'\t{wl_xfit[0]}')
f.write(f'\tsum of squared fit residuals: {wl_xfit[1][0][0]}\n')
f.close()

wl_wxfit = Polynomial.fit(x=nfw_gtan[wl], y=diff_gcross[wl], deg=1, full=True, w=1./xweight[wl])
y_wl_wxfit = wl_xfit[0].coef[0] + xx*wl_xfit[0].coef[1]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted PolyFit to gcross in gtan < {wl_regime} regime:\n##')
f.write(f'\t{wl_wxfit[0]}')
f.write(f'\tsum of squared fit residuals: {wl_wxfit[1][0][0]}\n')
f.close()

###
### For fun, try a second-order polyfit to the full gtan values
###
fit2 = Polynomial.fit(x=nfw_gtan, y=diff_gtan, deg=2, full=True)
y_fit2 = fit2[0].coef[0] + xx*fit2[0].coef[1] + xx*xx*fit2[0].coef[2]

f = open(aboutTxt, "a")
f.write(f'##\n##  2nd order PolyFit to all gtan values:\n##')
f.write(f'\t{fit2[0]}')
f.write(f'\tsum of squared fit residuals: {fit2[1][0][0]}\n')
f.close()

wfit2 = Polynomial.fit(x=nfw_gtan, y=diff_gtan, deg=2, full=True, w=1./weight)
y_wfit2 = wfit2[0].coef[0] + xx*wfit2[0].coef[1]+ xx*xx*wfit2[0].coef[2]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted 2nd order PolyFit to all gtan values:\n##')
f.write(f'\t{wfit2[0]}')
f.write(f'\tsum of squared fit residuals: {wfit2[1][0][0]}\n')
f.close()

wl_wfit2 = Polynomial.fit(x=nfw_gtan[wl], y=diff_gtan[wl], deg=2, full=True, w=1./weight[wl])
y_wl_wfit2 = wl_wfit2[0].coef[0] + xx*wl_wfit2[0].coef[1]+ xx*xx*wl_wfit2[0].coef[2]

f = open(aboutTxt, "a")
f.write(f'##\n##  Weighted PolyFit to gtan < {wl_regime}:\n##')
f.write(f'\t{wl_wfit2[0]}')
f.write(f'\tsum of squared fit residuals: {wl_wfit2[1][0][0]}\n')
f.close()

fig,ax = plt.subplots(1,1,figsize=(14,9))
ax.plot(nfw_gtan, stacked_shears['mean_gtan'], 'o', markersize=3)
xx = np.linspace(-1, 1)
ax.plot(xx, xx, '--k', alpha=0.7)
ax.set_xlabel(r'NFW $g_{\rm tan}$', fontsize=20)
ax.set_ylabel(r'meas $g_{\rm tan}$', fontsize=20)
ax.set_ylim(-0.1, 0.75)
ax.set_xlim(0.005, 0.4)
fig.tight_layout()
fig.savefig('gtan_meas_vs_nfw'+ dt_string +'.png')
plt.show()
fig,ax = plt.subplots(1,1,figsize=(14,9))

ax.plot(nfw_gtan, diff_gtan, 'o', markersize=3,alpha=0.5)
xx = np.linspace(-1, 1)
ax.axhline(0, linestyle='--', color='black', alpha=0.5)

yfit2_label = \
    r'$c = {cval:.5f}$\ \ \ \ $m_\gamma = {mval:.5f}$\ \ \ \ ${mgamma2} = {m2val:.5f}$'\
    .format(
        cval=wfit2[0].coef[0], \
        mval=wfit2[0].coef[1], \
        mgamma2 = r'm_{\gamma^2}', \
        m2val=wfit2[0].coef[2] \
        )
ax.plot(xx, y_wfit2, label=yfit2_label,alpha=0.8, color='darkorange')

yfit_label = \
    r'$c = {cval:.5f}$\ \ \ \ $m_\gamma = {mval:.5f}$'\
    .format(
        cval=wfit[0].coef[0], 
        mval=wfit[0].coef[1]
        )
ax.plot(xx, y_wfit, label=yfit_label, alpha=0.7, color='red')


ax.set_ylabel(r'NFW $g_{\rm tan}$ - meas $g_{\rm tan}$', fontsize=20)
ax.set_xlabel(r'meas $g_{\rm tan}$', fontsize=20)

ax.set_ylim(-0.2, 0.2)
ax.set_xlim(-0.005, 0.35)
ax.legend(fontsize=16)
fig.tight_layout()

fig.savefig('diff_gtan_meas_vs_nfw'+ dt_string +'.png')
fig,ax = plt.subplots(1,1,figsize=(14,9))
ax.plot(nfw_gtan[wl], diff_gcross[wl], '.',alpha=0.5, color='tab:orange')
ax.plot(nfw_gtan[wl], diff_gtan[wl], '.',alpha=1, color='tab:blue')

ax.axhline(0, linestyle='--', color='black',alpha=0.5)

xx = np.linspace(-1, 1)
yfit_label = \
    r'$c = {cval:.5f}$\ \ \ \ $m_\gamma = {mval:.5f}$'\
    .format(
        cval=wl_wfit[0].coef[0], 
        mval=wl_wfit[0].coef[1]
        )
ax.plot(xx, y_wl_wfit, label=yfit_label, alpha=0.7, c='tab:blue')

y_xfit_label = \
    r'$c = {cval:.5f}$\ \ \ \ $m_\gamma = {mval:.5f}$'\
    .format(
        cval=wl_wxfit[0].coef[0], 
        mval=wl_wxfit[0].coef[1]
        )
ax.plot(xx, y_wl_wxfit, label=y_xfit_label, alpha=0.7, lw=2, color='tab:orange')


ax.set_ylabel(r'meas $g_{\rm tan}$ - NFW $g_{\rm tan}$', fontsize=20)
ax.set_xlabel(r'NFW $g_{\rm tan}$', fontsize=20)
ax.set_xlim(0.005,1.1*wl_regime)
ax.legend(fontsize=20)
ax.set_xlim(min(stacked_shears['mean_gtan']),max(stacked_shears['mean_gtan']) )
fig.tight_layout()
fig.savefig('gx_gtan_diff_vs_nfw'+dt_string+'.png')
fig,ax = plt.subplots(1,1,figsize=(14,9))
ax.plot(nfw_gtan[wl], diff_gtan[wl], '.',alpha=0.8)
ax.axhline(0, linestyle='--', color='black',alpha=0.7)

xx = np.linspace(-1, 1)
yfit_label = \
    r'$c = {cval:.5f}$\ \ \ \ $m_\gamma = {mval:.5f}$'\
    .format(
        cval=wl_wfit[0].coef[0], 
        mval=wl_wfit[0].coef[1]
        )
ax.plot(xx, y_wl_wfit, label=yfit_label, alpha=0.7, color='red')

ax.set_ylabel(r'meas $g_{\rm tan}$ - NFW $g_{\rm tan}$', fontsize=20)
ax.set_xlabel(r'NFW $g_{\rm tan}$', fontsize=20)
ax.set_xlim(0.005,1.15*wl_regime)
ax.legend(fontsize=20)
#ax.set_xlim(min(stacked_shears['mean_gtan']),max(stacked_shears['mean_gtan']) )
fig.tight_layout()
fig.savefig('gtan_diff_vs_nfw'+dt_string+'.png')

