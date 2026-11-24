"""Compositional a-posteriori certificate for a labelled reduced AC response.

Real-arithmetic theorem: feeder certificates bound exact sigma, sigma' and
sigma'' over a normalized-load interval. Apply a Newton-preconditioned
contraction to the MAIN network with these error bounds. Its dimension does
not include feeder internal buses. Numerical guards are binary64 estimates,
not an outward-rounded interval arithmetic implementation.
"""
import numpy as np
from math import sqrt


def certify_main(reduced, solution, tubes):
    s=reduced.main;x=solution['main_x'];v=s.voltage(x)
    f,j=reduced.residual_jac(x)
    C=np.linalg.inv(j);AC=abs(C)
    ferr=np.full(s.n,2e-12)
    jerr=np.zeros_like(j)
    if not hasattr(s,'_certificate_hrow'):
        s._certificate_hrow=np.zeros(s.n)
        k,a,b,hh=s.hessian_entries()
        np.add.at(s._certificate_hrow,k,abs(hh))
    hrow=s._certificate_hrow.copy()
    zvals=[]
    for source,response,tube in zip(reduced.sources,reduced.responses,tubes):
        if not tube['certified']:
            return dict(certified=False,reason='uncertified_feeder')
        idx=int(np.where(s.ids==source)[0][0]);u=abs(v[source]);z=reduced.load/u**2
        zvals.append(z)
        if not tube['domain'][0]<=z<=tube['domain'][1]:
            return dict(certified=False,reason='outside_response_domain')
        e0=float(tube['power_error']);e1=float(tube['power_derivative_error'])
        for row in [idx]+([] if s.pv[idx] else [s.m+idx]):
            ferr[row]+=u*u*e0
            jerr[row,[idx,s.m+idx]]+=2*np.array([abs(v[source].real),abs(v[source].imag)])*(e0+z*e1)
            coeff=response.coeff[:,:2]
            sb=float(np.sum(np.hypot(coeff[:,0],coeff[:,1])))+e0
            dc=response.dcoeff[:,:2]
            sd=float(np.sum(np.hypot(dc[:,0],dc[:,1])))+e1
            zm=tube['domain'][1]
            hrow[row]+=4*(sb+zm*sd)+8*zm*zm*tube['power_second_derivative_bound']
    rho=float(np.max(abs(C@f)+AC@ferr))
    A=np.eye(s.n)-C@j
    # Roundoff guard includes the computed inverse defect rather than assuming it zero.
    aa=float(np.max(np.sum(abs(A)+AC@jerr,axis=1)))+2e-11
    h=float(np.max(AC@hrow))*(1+1e-11)
    if aa>=1:return dict(certified=False,reason='jacobian_defect',a=aa)
    R=max(1e-12,1.1*rho/(1-aa))
    for _ in range(12):
        if aa+h*R<1 and rho+aa*R+.5*h*R*R<=R:break
        R*=2
    else:return dict(certified=False,reason='no_self_mapping',rho=rho,a=aa,h=h)
    delta=sqrt(2)*R
    for source,tube in zip(reduced.sources,tubes):
        u=abs(v[source])
        if u<=delta or reduced.load/(u+delta)**2<tube['domain'][0] or reduced.load/(u-delta)**2>tube['domain'][1]:
            return dict(certified=False,reason='neighborhood_leaves_response_domain')
    E=min(R,rho/(1-aa-h*R))
    return dict(certified=True,radius=R,error=E,rho=rho,a=aa,h=h,contraction=aa+h*R,
                dimension=s.n,semantics='Local coupled high-voltage branch; real-arithmetic certificate with guarded floating implementation.')


def slack_loss_interval(reduced,solution,certificate):
    """Exact loss at the AC root equals total generator P minus fixed demand.

    Non-slack P is specified. The slack branch-injection polynomial depends on
    main voltages only because both feeder attachment buses are non-slack.
    This avoids the ill-scaled dense feeder loss Hessian.
    """
    if not certificate['certified']:return (-np.inf,np.inf)
    s=reduced.main;v=s.voltage(solution['main_x']);sl=s.slack
    if sl in reduced.sources:raise NotImplementedError('Slack-attached feeder')
    p=s.ppc;gen=p['gen'];slack_generation=(v[sl]*np.conj(s.Y[sl]@v)).real*s.base+p['bus'][sl,2]
    fixed_p=float(gen[(gen[:,0].astype(int)-1)!=sl,1].sum())
    demand=float(reduced.full.ppc['bus'][:,2].sum())
    nominal=slack_generation+fixed_p-demand
    # The slack voltage is fixed, making this output linear in non-slack V.
    row=v[sl]*np.conj(s.Y[sl,s.ids])*s.base
    err=float((abs(row.real).sum()+abs(row.imag).sum())*certificate['error'])+2e-10
    return (nominal-err,nominal+err)


def classify(reduced,solution,certificate,tubes,tol=1e-7):
    """Bounds voltage, branch-MVA, and generator limits for the same AC root.

    Feeders use normalized current response bounds rather than |Y| voltage
    propagation, which would amplify harmless voltage uncertainty on tiny-Z
    branches. Any unresolved constraint triggers exact evaluation.
    """
    if not certificate['certified']:return dict(status='unresolved')
    full=reduced.full;main=reduced.main;v=solution['voltage'];R=certificate['error'];d=sqrt(2)*R
    verr=np.zeros(full.nb);verr[:main.nb]=d;verr[main.slack]=0
    # Main-branch currents do not include feeder injections.
    branch=full.ppc['branch'];nbmain=len(main.ppc['branch'])
    iferr=np.zeros(len(branch));iterr=np.zeros(len(branch))
    iferr[:nbmain]=abs(main.Yf)@verr[:main.nb]
    iterr[:nbmain]=abs(main.Yt)@verr[:main.nb]
    feed_power_errors=[]
    for k,(source,response,tube) in enumerate(zip(reduced.sources,reduced.responses,tubes)):
        u=abs(v[source]);z=reduced.load/u**2
        dz=max(abs(reduced.load/(u+d)**2-z),abs(reduced.load/(u-d)**2-z))
        y,dy=response.evaluate(z);m=response.system.m;wh=y[2:2+m]+1j*y[2+m:]
        ew=np.asarray(tube['voltage_error']);dw=np.asarray(tube['voltage_derivative_bound'])
        fm=full.meta['feeders'][k];start=fm['start'];count=len(fm['active'])
        verr[start:start+count]=d*abs(wh)+(u+d)*(ew+dw*dz)+2e-12
        localv=np.r_[1+0j,wh]
        for mat,current,out in [(response.system.Yf,tube['current_from'],iferr),
                                (response.system.Yt,tube['current_to'],iterr)]:
            val=mat@localv
            e=np.asarray(current['error']);der=np.asarray(current['derivative_bound'])
            ri=fm['tr_index'];out[ri:ri+len(val)]=d*abs(val)+(u+d)*(e+der*dz)+2e-12
        coeff=response.dcoeff[:,:2]
        dsigma=float(np.sum(np.hypot(coeff[:,0],coeff[:,1])))+tube['power_derivative_error']
        sig=complex(y[0],y[1]);esig=tube['power_error']+dsigma*dz
        feed_power_errors.append((source,(2*u*d+d*d)*abs(sig)+(u+d)**2*esig))
    magnitude=abs(v);vlo=magnitude-verr;vhi=magnitude+verr
    ilf=full.Yf@v;ilt=full.Yt@v
    bi=branch[:,0].astype(int)-1;bj=branch[:,1].astype(int)-1
    sf=abs(v[bi]*np.conj(ilf))*full.base;st=abs(v[bj]*np.conj(ilt))*full.base
    sef=(abs(v[bi])*iferr+abs(ilf)*verr[bi]+verr[bi]*iferr)*full.base
    set_=(abs(v[bj])*iterr+abs(ilt)*verr[bj]+verr[bj]*iterr)*full.base
    rating=branch[:,5];valid=rating>0
    # Generator injection errors are computed on the small main network plus
    # feeder boundary power errors, not from full-network admittance rows.
    vm=v[:main.nb];im=main.Y@vm;imerr=abs(main.Y)@verr[:main.nb]
    genserr=(abs(vm)*imerr+abs(im)*verr[:main.nb]+verr[:main.nb]*imerr)*main.base
    for source,e in feed_power_errors:genserr[source]+=e*main.base
    injection=v*np.conj(full.Y@v)*full.base
    generation=injection+full.ppc['bus'][:,2]+1j*full.ppc['bus'][:,3]
    lower=[float((.9-vhi.min())/.1),float((vlo.max()-1.1)/.1)]
    upper=[float((.9-vlo.min())/.1),float((vhi.max()-1.1)/.1)]
    lower.append(float(np.max(np.maximum(sf[valid]-sef[valid],st[valid]-set_[valid])/rating[valid]-1)))
    upper.append(float(np.max(np.maximum(sf[valid]+sef[valid],st[valid]+set_[valid])/rating[valid]-1)))
    for g in full.ppc['gen']:
        b=int(g[0])-1;val=generation[b];e=genserr[b]
        for nominal,scale in [(val.real-g[8],max(1,g[8])),(g[9]-val.real,max(1,g[8])),(val.imag-g[3],max(1,abs(g[3]))),(g[4]-val.imag,max(1,abs(g[4])))]:
            lower.append(float((nominal-e)/scale));upper.append(float((nominal+e)/scale))
    lo=max(lower);hi=max(upper)
    status='feasible' if hi<=tol else 'violating' if lo>tol else 'unresolved'
    return dict(status=status,violation_lower=lo,violation_upper=hi,max_voltage_error=float(verr.max()))
