"""Global parameter-domain certificates for an existing feeder Chebyshev curve.

The data-independent proof uses polynomial coefficient bounds, not validation
samples. For a fixed all-PQ feeder label, Yw+y0+z*conj(d)/conj(w)=0.
The supplied polynomial wh(z) is surrounded by a componentwise complex-disc
tube, uniformly for z in the complete declared interval. The computed inverse
defect is retained. No planner reference state is used in construction.

The scalar homogeneous normalization and validated polynomial approximation
are established numerical ideas. This module makes no standalone novelty claim.
"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'theory'))
sys.path.insert(0,str(ROOT/'research'/'normalized_response'))
import ac
import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.linalg import solve
from certificates import gamma,up


def l1(coeff):return np.sum(np.abs(coeff),axis=0)

def pad(a,n):
    out=np.zeros(n,dtype=complex);out[:len(a)]=a;return out

def _complex_dot(A,B):
    value=A@B
    rad=gamma(8*A.shape[-1]+24)*(np.abs(A)@np.abs(B))
    return value,up(rad+1e-300)

def _positive_solve(B,b):
    """Solve an M-matrix comparison with a posteriori inverse-defect guard."""
    n=len(B);M=np.eye(n)-B
    value=solve(M,b,check_finite=False)
    # Scalar inflation is used only after a componentwise residual check.
    value=np.maximum(value,0.)*(1+1e-10)+1e-15
    for _ in range(12):
        need=b+B@value
        if np.all(need<=value):return up(value)
        value=np.maximum(value,need)*(1+1e-8)+1e-15
    raise ValueError('Positive derivative majorant did not close.')


def certify_response(response,max_radius_steps=100):
    start=time.perf_counter();s=response.system;ids=s.ids
    if np.any(s.pv):raise ValueError('The feeder tube requires an all-PQ feeder.')
    if abs(s.vs-1)>1e-14:raise ValueError('Normalize the source phasor to one.')
    domain=tuple(response.domain);zmax=max(abs(domain[0]),abs(domain[1]))
    n=s.m;degree=len(response.coeff)-1
    W=response.coeff[:,2:2+n]+1j*response.coeff[:,2+n:]
    P=ch.chebder(W,axis=0)*2/(domain[1]-domain[0])
    zcoeff=np.array([sum(domain)/2,(domain[1]-domain[0])/2])
    Y=s.Y[np.ix_(ids,ids)];y0=s.Y[ids,s.slack]*s.vs
    demand=-s.spec[ids];dbar=np.conj(demand)
    Z=np.linalg.inv(Y);M=Z*dbar[None,:]
    ZY,ZYrad=_complex_dot(Z,Y)
    Dabs=np.abs(np.eye(n)-ZY)+ZYrad+gamma(4)*(np.eye(n)+np.abs(ZY))
    Mab=np.abs(M)*(1+gamma(8));Zab=np.abs(Z)
    L,Lrad=_complex_dot(Y,W.T);L=L.T;Lrad=Lrad.T;L[0]+=y0
    Lrad[0]+=gamma(4)*(np.abs(L[0])+np.abs(y0))
    LP,LPrad=_complex_dot(Y,P.T);LP=LP.T;LPrad=LPrad.T
    Wabs=l1(W);Pabs=l1(P)
    mu=np.abs(W[0])-l1(W[1:])
    mu=np.nextafter(mu*(1-gamma(4*degree+16)),-np.inf)
    output=dict(certified=False,domain=domain,degree=degree,min_polynomial_voltage=float(min(mu)),
                seconds=0.)
    if min(mu)<=0:output.update(reason='polynomial_disk_contains_zero',seconds=time.perf_counter()-start);return output
    residual=np.zeros(n);derivative_residual=np.zeros(n)
    for k in range(n):
        r=pad(ch.chebmul(np.conj(W[:,k]),L[:,k]),2*degree+1)
        r[:2]+=zcoeff*dbar[k]
        formation=gamma(4*(degree+1)**2+48)*(Wabs[k]*l1(L[:,k])+zmax*abs(dbar[k]))
        formation+=Wabs[k]*np.sum(Lrad[:,k])
        residual[k]=np.sum(abs(r))+formation
        # G=conj(wh)^2 Y wh' + conj(d)*(conj(wh)-z conj(wh')).
        first=ch.chebmul(ch.chebmul(np.conj(W[:,k]),np.conj(W[:,k])),LP[:,k])
        second=pad(np.conj(W[:,k]),max(len(W),len(P)+1))-pad(ch.chebmul(zcoeff,np.conj(P[:,k])),max(len(W),len(P)+1))
        g=pad(first,max(len(first),len(second)))+dbar[k]*pad(second,max(len(first),len(second)))
        gformation=gamma(12*(degree+1)**2+80)*(Wabs[k]**2*l1(LP[:,k])+abs(dbar[k])*(Wabs[k]+zmax*Pabs[k]))
        gformation+=Wabs[k]**2*np.sum(LPrad[:,k])
        derivative_residual[k]=np.sum(abs(g))+gformation
    residual=up(residual*(1+gamma(8*degree+20)))
    derivative_residual=up(derivative_residual*(1+gamma(12*degree+30)))
    rho=up((Zab@(residual/mu))*(1+gamma(4*n+20)))
    B0=Dabs+zmax*Mab/(mu[None,:]**2)
    try:
        r=_positive_solve(B0,rho)*1.05
    except Exception:
        output.update(reason='linear_comparison_failed',seconds=time.perf_counter()-start);return output
    for iteration in range(max_radius_steps):
        vmin=mu-r
        if min(vmin)<=0:
            output.update(reason='tube_contains_zero',seconds=time.perf_counter()-start);return output
        B=Dabs+zmax*Mab/(vmin[None,:]**2)
        q=float(max((B@r)/r))
        need=rho+Dabs@r+zmax*Mab@(r/(mu*vmin))
        need=up(need*(1+gamma(4*n+24)))
        if q>=1:
            output.update(reason='noncontractive_tube',q=q,seconds=time.perf_counter()-start);return output
        if np.all(need<=r):break
        r=np.maximum(r,need)*1.05
    else:
        output.update(reason='inclusion_not_closed',seconds=time.perf_counter()-start);return output
    # Exact derivative bounds for the unique feeder response over the domain.
    d1=_positive_solve(B,Mab@(1/vmin))
    d2=_positive_solve(B,2*Mab@(d1/vmin**2)+2*zmax*Mab@(d1*d1/vmin**3))
    derivative_rhs=Zab@(derivative_residual/mu**2)
    derivative_rhs+=Mab@(r/(mu*vmin))+2*zmax*Mab@(Pabs*r/vmin**3)
    derivative_error=_positive_solve(B,derivative_rhs)
    # Use current-balance substitution for terminal outputs; keep inverse defect.
    def current_outputs(A,constant):
        K,Kr=_complex_dot(A,Z)
        KY,KYr=_complex_dot(K,Y)
        defect=np.abs(A-KY)+KYr+gamma(4)*(np.abs(A)+np.abs(KY))
        Ka=np.abs(K)
        curr_coeff,curr_coeff_rad=_complex_dot(A,W.T)
        curr_coeff=curr_coeff.T;curr_coeff_rad=curr_coeff_rad.T
        curr_coeff[0]+=constant
        formation=np.sum(curr_coeff_rad,axis=0)+gamma(4)*(np.abs(curr_coeff[0])+np.abs(constant))
        evaluation=gamma(8*degree+24)*l1(curr_coeff)
        fbound=residual/mu+zmax*np.abs(demand)*r/(mu*vmin)
        error=Ka@fbound+defect@r+formation+evaluation
        deriv_bound=Ka@(np.abs(demand)/vmin+zmax*np.abs(demand)*d1/vmin**2)+defect@d1
        deriv_change=(derivative_residual/mu**2+np.abs(demand)*r/(mu*vmin)
                      +zmax*np.abs(demand)*(derivative_error/vmin**2+2*Pabs*r/vmin**3))
        deriv_error=Ka@deriv_change+defect@derivative_error
        second_bound=Ka@(2*np.abs(demand)*d1/vmin**2+zmax*np.abs(demand)*d2/vmin**2
                         +2*zmax*np.abs(demand)*d1*d1/vmin**3)+defect@d2
        return dict(coeff=curr_coeff,error=up(error),derivative_bound=up(deriv_bound),
                    derivative_error=up(deriv_error),second_derivative_bound=up(second_bound))
    source=current_outputs(s.Y[s.slack:s.slack+1,ids],np.array([s.Y[s.slack,s.slack]*s.vs]))
    sigma_poly=response.coeff[:,0]+1j*response.coeff[:,1]
    sigma_exact_coeff=np.conj(source['coeff'][:,0])
    sigma_inconsistency=float(np.sum(abs(sigma_poly-sigma_exact_coeff)))
    sigmad_poly=ch.chebder(sigma_poly)*2/(domain[1]-domain[0])
    sigmad_exact=ch.chebder(sigma_exact_coeff)*2/(domain[1]-domain[0])
    sigma_derivative_inconsistency=float(np.sum(abs(sigmad_poly-sigmad_exact)))
    source_error=float(source['error'][0])+sigma_inconsistency
    source_derror=float(source['derivative_error'][0])+sigma_derivative_inconsistency
    source_derror+=gamma(8*degree+24)*float(np.sum(abs(sigmad_poly)))
    terminal_from=current_outputs(s.Yf[:,ids],s.Yf[:,s.slack]*s.vs)
    terminal_to=current_outputs(s.Yt[:,ids],s.Yt[:,s.slack]*s.vs)
    output.update(certified=True,voltage_error=r,voltage_derivative_error=derivative_error,
                  voltage_derivative_bound=d1,voltage_second_derivative_bound=d2,
                  polynomial_voltage_lower=mu,exact_voltage_lower=vmin,
                  power_error=source_error,power_derivative_error=source_derror,
                  power_derivative_bound=float(source['derivative_bound'][0]),
                  power_second_derivative_bound=float(source['second_derivative_bound'][0]),
                  current_from=terminal_from,current_to=terminal_to,
                  branch_current_error_f=terminal_from['error'],
                  branch_current_error_t=terminal_to['error'],
                  branch_current_derivative_bound_f=terminal_from['derivative_bound'],
                  branch_current_derivative_bound_t=terminal_to['derivative_bound'],
                  q=q,radius_iterations=iteration+1,power_polynomial_inconsistency=sigma_inconsistency,
                  seconds=time.perf_counter()-start)
    return output


def serializable(value):
    """JSON representation; complex coefficient arrays retain real/imag parts."""
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value):return dict(real=value.real.tolist(),imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def test():
    from atlas import Response,feeder_at
    rows=[]
    for feeder in (0,1):
      for degree in (6,10):
       for X,F,tap,cap in [(0.,0,1.04,0.),(.5,1,1.075,1.2),(1.,1,1.1,1.2)]:
        response=Response(feeder_at(feeder,X,F,tap,cap),degree)
        cert=certify_response(response)
        row=dict(feeder=feeder,degree=degree,X=X,F=F,tap=tap,cap=cap,
                 certified=cert['certified'],training_seconds=response.seconds,certificate_seconds=cert['seconds'])
        if cert['certified']:
            row.update(power_error=cert['power_error'],power_derivative_error=cert['power_derivative_error'],
                       max_voltage_error=float(max(cert['voltage_error'])),q=cert['q'],
                       power_second_derivative_bound=cert['power_second_derivative_bound'])
            # Independent diagnostic points do not enter construction.
            original=response.system.spec.copy();maxv=0.;maxp=0.;coverage=True
            for z in np.linspace(.4,1.8,19):
                response.system.spec=original*z;truth=response.system.solve()
                if not truth['converged']:raise AssertionError('Reference solver failed')
                y,dy=response.evaluate(z);wp=y[2:2+response.system.m]+1j*y[2+response.system.m:]
                ev=abs(truth['voltage'][response.system.ids]-wp)
                power=np.conj(response.system.Y[0]@truth['voltage']);ep=abs(power-(y[0]+1j*y[1]))
                maxv=max(maxv,max(ev));maxp=max(maxp,ep)
                coverage &=bool(np.all(ev<=cert['voltage_error']+1e-10) and ep<=cert['power_error']+1e-10)
            response.system.spec=original
            row.update(observed_max_voltage_error=float(maxv),observed_max_power_error=float(maxp),covered=coverage)
            assert coverage,row
        else:row['reason']=cert.get('reason')
        rows.append(row);print(json.dumps(row),flush=True)
    Path(__file__).with_name('feeder_tube_pilot.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':test()
