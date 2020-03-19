import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy import integrate, optimize

#-----------------------------------------------------------------------------%
# PARAMETERS                                                                  %
#-----------------------------------------------------------------------------%

# code parameters
n_plot = 200
names = {'bln' : 'naive', 'myo' : 'myopic', 'rat' : 'rational'}

# primitives
r = .04
f = 4
lam = 2.5 

# prics
p1 = f-1.
p0 = .8*p1

L = .95*np.sqrt(p1/p0)
N = 4.
pi = 1./N
C = 10.
m = .5

# the greeks
alp = r/(r+lam)
bet = (lam*L)/(C*N**2.)

# derived
v1_myo = 0.
v1_rat = (1.-alp)*pi*(1.-m) 
a1 = v1_rat+alp*f+(1.-alp)*p1 
b1 = -alp
V1 = C*np.log((p1-p0)/(L*p0-p0))/L # cum volume at the time of adjustment

# naive model adjustment time
t1_bln = np.log((p1-p0)/(L*p0-p0))/(bet*(N-1.)) 

# parameter diagnostics
L1 = np.log((p1-p0)/(L*p0-p0))
if L1 > L:
    print('warning: first comp after shock')

if L*p0 > p1:
    print('warning: price automatically adjusts')

#-----------------------------------------------------------------------------#
# RIGHT-HAND-SIDES                                                            #  
#-----------------------------------------------------------------------------#

# BLIND
def rhs_bln(t1,t,y):
    """capitals denote derivatives"""
    _, p = y
    V = lam*(N-1.)/N**2.
    P = bet*(p-p0)*(N-1.)
    return np.array([V,P])

# MYOPIC
def rhs_myo(t1,t,y):
    """capitals denote derivatives"""

    # parameters (find a better way of doing this)
    A_lim = -r 
    B_lim = r 
    m_lim = np.array([[bet*(p1-p0),1.],[1.,-1./r]])  
    v_lim = np.array([bet*(p1-p0)*N,1.]) 
    x_lim, P_lim = la.solve(m_lim,v_lim)
    e_lim = (1.-np.exp(-lam*t1))/N
    e_tld_lim = e_lim+(1.-e_lim)*(x_lim-1.)/(N-1.)

    # output 
    if abs(t-t1)<1.e-10:
        A = A_lim 
        B = B_lim 
        V = lam*(N-x_lim)/N**2.
        E = V/e_tld_lim 
        P = P_lim
    else:
        a, b, _,  _, p = y
        v = v1_myo*np.exp(-r*(t1-t))+p
        x = (v-a)/b
        A = r*a-r*f
        B = r*b+r 
        V = lam*(N-x)/N**2.
        e = (1.-np.exp(-lam*t))/N
        e_tld = e+(1.-e)*(x-1.)/(N-1.)
        E = V/e_tld
        P = bet*(p-p0)*(N-x)

    return np.array([A,B,E,V,P])

# RATIONAL  
def rhs_rat(t1,t,y):
	"""capitals denote derivatives"""

	# parameters (find a better way of doing this)
	F_lim = pi*((v1_rat+alp*f+(1.-alp)*p1)-m*alp)+pi*(p1+v1_rat)*(N-1.)
	Au_lim = (r+lam)*(v1_rat+p1)-r*f-lam*F_lim 
	Bu_lim = r
	Ad_lim = (r+lam)*(v1_rat+alp*f+(1.-alp)*p1)-r*f-lam*F_lim 
	Bd_lim = 0.
	m_lim = np.array([[bet*(p1-p0),1.],[1.,-1./r]])  
	v_lim = np.array([bet*(p1-p0)*N,v1_rat-(v1_rat+p1)/alp+f+(lam/r)*F_lim]) 
	x_lim, P_lim = la.solve(m_lim,v_lim)
		
	# output
	if t == t1:
		Au = Au_lim 
		Bu = Bu_lim 
		Ad = Ad_lim 
		Bd = Bd_lim 
		V = lam*(N-x_lim)/N**2.
		P = P_lim
	else:
		au, bu, ad, bd, _, p = y
		v = v1_rat*np.exp(-r*(t1-t))+p
		x = (v-au)/bu
		F = pi*(ad+bd*m)+pi*(x-1.)*au+.5*pi*(x**2.-1.)*bu+pi*v*(N-x)
		Au = (r+lam)*au-r*f-lam*F
		Bu = (r+lam)*bu+r 
		Ad = (r+lam)*ad-r*f-lam*F
		Bd = (r+lam)*bd+r 
		V = lam*(N-x)/N**2.
		P = bet*(p-p0)*(N-x)

	return np.array([Au,Bu,Ad,Bd,V,P])

#-----------------------------------------------------------------------------#
# PLOTTER                                                                     #  
#-----------------------------------------------------------------------------#

# plotter
def plotter(sols,times,values,linesty,fillcol):

    # plots
    t_max = 1.25*max(times.values())
    p_min, p_max = p0-.25*(p1-p0), p1+.25*(p1-p0)
    t_plot = np.linspace(0.,t_max,n_plot)
    p_plot = np.linspace(p_min,p_max,n_plot)

    # prices -----------------------------------------------------------------#
    xticks = [0.]
    xticklabels = ['$0.$']
    for sol in sols:
        t_plot_up = np.linspace(times[sol],1.2*t_max,n_plot)
        tt = np.hstack([np.flip(sols[sol].t),t_plot_up])
        yy = np.hstack([np.flip(sols[sol].y[-1]),p1*np.ones(n_plot)])
        plt.plot(tt,yy,linesty[sol],linewidth=2)
        xticks += [times[sol]]
        xticklabels += ['$t_{1}$']
    t_plot = np.linspace(-.20*t_max,0.,n_plot)
    plt.plot(t_plot,p0+0.*t_plot,'-k',linewidth=2)
    t_plot = np.linspace(t_max,1.20*t_max,n_plot)
    plt.plot(t_plot,p1+0.*t_plot,'-k',linewidth=2)
    plt.axis([-.20*t_max,1.20*t_max,p_min,p_max])
    plt.legend(['naive','myopic','rational'],loc='lower right')
    plt.xlabel('time $(t)$')
    plt.ylabel('price $(p)$')
    plt.xticks(xticks,xticklabels)
    plt.yticks([p0,L*p0,p1],['$p_{0}$','$Lp_{0}$','$p_{1}$'])
    plt.grid()
    plt.title('the price path $p(t)$')
    plt.savefig('price.pdf')
    plt.close()

    # indifferent type -------------------------------------------------------#
    xticks = [0.]
    xticklabels = ['$0.$']
    legs = []
    for sol in sols:
        plt.plot(0.,0.,linesty[sol],linewidth=2)
    for sol in sols:
        if True: 
            if len(sols[sol].y)>2:
                vv = values[sol]*np.exp(-r*(times[sol]-sols[sol].t))+sols[sol].y[-1]
                xx = (vv-sols[sol].y[0])/sols[sol].y[1]
                plt.fill_between(sols[sol].t,np.ones(n_plot),xx,facecolor=fillcol[sol],interpolate=True,alpha=.75)
            else:
                xx = np.ones(n_plot)
            t_plot_dn = np.linspace(-.20*t_max,0.,n_plot)
            t_plot_up = np.linspace(times[sol],1.20*t_max,n_plot)
            y_plot = np.ones(n_plot)
            tt = np.hstack([t_plot_dn,np.flip(sols[sol].t),t_plot_up])
            yy = np.hstack([y_plot,np.flip(xx),y_plot])
            plt.plot(t_plot_dn,np.ones(n_plot),linesty[sol],linewidth=2)
            plt.plot(sols[sol].t,xx,linesty[sol],linewidth=2)
            plt.plot(t_plot_up,np.ones(n_plot),linesty[sol],linewidth=2)
            xticks += [times[sol]]
            xticklabels += ['$t_{1}^{' + names[sol] + '}$']
            legs += [names[sol]]
    # pre-post
    plt.axis([-.20*t_max,1.20*t_max,-1./N,N+1./N])
    plt.legend(legs,loc='lower right')
    plt.xlabel('time $(t)$')
    plt.ylabel('disutility',usetex=True)
    plt.title('the indifferent type')
    plt.annotate('owner waits to sell',(.05*t1_bln,1.3))
    plt.annotate('non-owner buys (to flip)',(.05*t1_bln,1.1))
    plt.annotate('owner doesn\'t sell',(.05*t1_bln,.4))
    plt.annotate('non-owner buys immediately',(.05*t1_bln,.2))
    plt.annotate('owner sells immediately',(.05*t1_bln,.8*N))
    plt.annotate('non-owner doesn\'t buy',(.05*t1_bln,.8*N-.2))
    plt.annotate('$\\overline{x}^{rational}(t)$',(-.19*t_max,2.1),usetex=True)
    plt.annotate('$\\overline{x}^{myopic}(t)$',(-.19*t_max,1.55),usetex=True)
    plt.xticks(xticks,xticklabels,usetex=True)
    plt.yticks([0.,1.,N],['$0$','$1$','$N$'])
    plt.grid()
    plt.savefig('indifference.pdf')
    plt.close()

#-----------------------------------------------------------------------------#
# HUNTER                                                                      #  
#-----------------------------------------------------------------------------#

# time hunter
def timehunt(t1,rhs,Y0,retsol):
        
    # rhs wrapper
    def rhs_wrap(t,y):
        return rhs(t1,t,y)

    # plot grid
    t_plot_hunt = np.linspace(t1,0.,n_plot)

    # rk45
    s = integrate.solve_ivp(rhs_wrap,
        t_span = (t1,0.),
        y0 = Y0,
        rtol = 1.e-5,
        t_eval = t_plot_hunt) 
    #plotter({'tmp': s},{'tmp' : t1}) 

    # output
    if retsol:
        return s
    else:
        return s.y[-1][-1]-L*p0

#-----------------------------------------------------------------------------#
# SOLUTIONS                                                                   #  
#-----------------------------------------------------------------------------#

# naive solution
print("naive solution")
Y0 = np.array([V1,p1])
t = {"bln" : t1_bln}
v = {"bln" : np.nan}
s = {"bln" : timehunt(t1_bln,rhs_bln,Y0,True)}
l = {"bln" : ':k'} 
fillcol = {"bln" : ''} 

# myopic solution
print("myopic solution")
Y0 = np.array([p1,0.,0.,V1,p1])
t1_myo = optimize.root_scalar(
        lambda t1: timehunt(t1,rhs_myo,Y0,False),
        x0 = t1_bln,
        x1 = 2.*t1_bln).root
t["myo"] = t1_myo
v["myo"] = v1_myo
s["myo"] = timehunt(t1_myo,rhs_myo,Y0,True)
l["myo"] = '--k'
fillcol["myo"] = 'black'

# rational solution
print("rational solution")
Y0 = np.array([v1_rat+p1,0.,a1,b1,V1,p1])
t1_rat = optimize.root_scalar(
        lambda t1: timehunt(t1,rhs_rat,Y0,False),
        x0 = t1_bln,
        x1 = 4.*t1_bln).root
t["rat"] = t1_rat 
v["rat"] = v1_rat
s["rat"] = timehunt(t1_rat,rhs_rat,Y0,True)
l["rat"] = '-k'
fillcol["rat"] = 'lightgray'

plotter(s,t,v,l,fillcol)
