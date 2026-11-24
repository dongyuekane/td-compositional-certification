"""Analytic certificates for quadratic AC equations and investment differences.

The model is the polynomial defined by the supplied binary64 coefficients.
H is a scipy sparse matrix of shape (n, n*n), with H[k,a*n+b] the
Hessian entry d2 F_k / dx_a dx_b. Both symmetric entries must be present.
No solved reference state or sampled derivative enters a certificate.

All norm estimates include conservative standard floating point gamma_n
guards. IEEE round-to-nearest, no overflow/underflow, and ordinary stable BLAS
arithmetic are assumed; this is not an interval-library formal verification.
The real-arithmetic inclusion and contraction conditions are exact.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
_vendor=Path(__file__).resolve().parents[1]/'vendor'
if _vendor.exists() and str(_vendor) not in sys.path:sys.path.append(str(_vendor))
import numpy as np
from scipy import sparse

U = np.finfo(float).eps / 2

def gamma(k):
    k = max(1, int(k))
    return k*U/(1-k*U)

def up(x):
    return np.nextafter(np.asarray(x, dtype=float), np.inf)

def abs_sum(x, axis=None):
    a = np.asarray(x)
    count = a.size if axis is None else a.shape[axis]
    return up(np.sum(np.abs(a), axis=axis)*(1+gamma(count+2)))

def norm_inf(a, radius=None):
    aa = np.asarray(a)
    if radius is not None:
        aa = np.abs(aa)+np.asarray(radius)
    if aa.ndim == 1:
        return float(up(np.max(np.abs(aa), initial=0.)))
    return float(up(np.max(abs_sum(aa, axis=1), initial=0.)))

def dot_enclosure(a, b, b_radius=None):
    a=np.asarray(a); b=np.asarray(b)
    out=a@b
    scale=np.abs(a)@np.abs(b)
    rad=gamma(a.shape[-1]+4)*scale
    if b_radius is not None:
        extra=np.abs(a)@np.asarray(b_radius)
        rad += extra*(1+gamma(a.shape[-1]+4))
    return out, up(rad*(1+gamma(8))+1e-300)

@dataclass
class QuadraticModel:
    constant: np.ndarray
    linear: np.ndarray
    H: object

    def __post_init__(self):
        self.constant=np.asarray(self.constant,dtype=float)
        self.linear=np.asarray(self.linear,dtype=float)
        self.n=len(self.constant)
        if sparse.issparse(self.H):
            self.H=sparse.csr_matrix(self.H,shape=(self.n,self.n*self.n))
        else:
            self.H=sparse.csr_matrix(np.asarray(self.H).reshape(self.n,-1))
        self.H.sum_duplicates(); self.H.eliminate_zeros(); self.H.sort_indices()
        coo=self.H.tocoo()
        self._eq=coo.row; self._a=coo.col//self.n; self._b=coo.col%self.n; self._hv=coo.data
        self._count=np.bincount(self._eq,minlength=self.n)

    @classmethod
    def from_system(cls, system):
        residual=getattr(system,'residual',None) or getattr(system,'F')
        jac=getattr(system,'jacobian',None) or getattr(system,'jac')
        n=getattr(system,'n',None) or getattr(system,'nvar')
        hh=system.hessian_entries()
        if isinstance(hh,tuple):
            eq,a,b,val=hh
            hh=sparse.coo_matrix((val,(eq,np.asarray(a)*n+b)),shape=(n,n*n))
        return cls(residual(np.zeros(n)),jac(np.zeros(n)),hh)

    def value_enclosure(self,x):
        x=np.asarray(x)
        lin,linrad=dot_enclosure(self.linear,x)
        terms=.5*self._hv*x[self._a]*x[self._b]
        quad=np.bincount(self._eq,weights=terms,minlength=self.n)
        scale=np.bincount(self._eq,weights=np.abs(terms),minlength=self.n)
        qrad=gamma(int(max(self._count,default=0))+8)*scale
        out=self.constant+lin+quad
        rad=linrad+qrad+gamma(5)*(np.abs(self.constant)+np.abs(lin)+np.abs(quad))
        return out,up(rad*(1+gamma(10))+1e-300)

    def jacobian_enclosure(self,x):
        x=np.asarray(x)
        ix=self._eq*self.n+self._a
        terms=self._hv*x[self._b]
        j=np.bincount(ix,weights=terms,minlength=self.n*self.n).reshape(self.n,self.n)
        scale=np.bincount(ix,weights=np.abs(terms),minlength=self.n*self.n).reshape(self.n,self.n)
        counts=np.bincount(ix,minlength=self.n*self.n)
        rad=gamma(int(max(counts,default=0))+5)*scale+gamma(3)*(np.abs(self.linear)+np.abs(j))
        return self.linear+j,up(rad*(1+gamma(8))+1e-300)

    def value(self,x): return self.value_enclosure(x)[0]
    def jacobian(self,x): return self.jacobian_enclosure(x)[0]

def transformed_hessian_norms(C, model_i, model_j=None, block_size=1024, C_j=None):
    """Bounds max_k sum_ab |(C H_i)[k,a,b]|, or C(H_i +/- H_j).

    Sparse multiplication precedes absolute values, retaining cancellation in
    the preconditioner and between two physical models. Memory is O(n*block).
    """
    C=np.asarray(C); n=model_i.n; ca=np.abs(C)
    Cj=C if C_j is None else np.asarray(C_j); caj=np.abs(Cj)
    total_plus=np.zeros(n); total_minus=np.zeros(n)
    active=np.unique(model_i.H.indices if model_j is None else np.r_[model_i.H.indices,model_j.H.indices])
    for start in range(0,len(active),block_size):
        cols=active[start:start+block_size]
        hi=model_i.H[:,cols]
        vi=np.asarray(hi.T@C.T).T
        si=np.asarray(abs(hi).T@ca.T).T
        ei=gamma(n+4)*si
        if model_j is None:
            plus=np.abs(vi)+ei
            total_plus+=np.sum(plus,axis=1)*(1+gamma(len(cols)+4))
        else:
            hj=model_j.H[:,cols]
            vj=np.asarray(hj.T@Cj.T).T
            sj=np.asarray(abs(hj).T@caj.T).T
            ej=gamma(n+4)*sj
            er=ei+ej+gamma(3)*(np.abs(vi)+np.abs(vj))
            total_plus+=np.sum(np.abs(vi+vj)+er,axis=1)*(1+gamma(len(cols)+8))
            total_minus+=np.sum(np.abs(vi-vj)+er,axis=1)*(1+gamma(len(cols)+8))
    guard=1+gamma(len(active)//block_size+8)
    if model_j is None:
        return float(up(np.max(total_plus,initial=0)*guard))
    return float(up(np.max(total_plus,initial=0)*guard)),float(up(np.max(total_minus,initial=0)*guard))

def _preconditioned(model,x,C):
    f,fr=model.value_enclosure(x); j,jr=model.jacobian_enclosure(x)
    r,rr=dot_enclosure(-C,f,fr)
    cj,cjr=dot_enclosure(C,j,jr)
    A=np.eye(model.n)-cj
    Ar=cjr+gamma(3)*(np.eye(model.n)+np.abs(cj))
    return r,rr,A,up(Ar)

def _linear_combine(a,ar,b,br,sign=1.,factor=1.):
    out=(a+sign*b)*factor
    rad=(ar+br+gamma(4)*(np.abs(a)+np.abs(b)))*abs(factor)
    return out,up(rad*(1+gamma(4)))

def single_certificate(model,x,C=None,hessian_norm=None,inflation=1.05):
    x=np.asarray(x)
    if C is None: C=np.linalg.inv(model.jacobian(x))
    r,rr,A,Ar=_preconditioned(model,x,C)
    rn=norm_inf(r,rr); a=norm_inf(A,Ar)
    h=transformed_hessian_norms(C,model) if hessian_norm is None else float(hessian_norm)
    out=dict(certified=False,residual_bound=rn,linear_bound=a,hessian_bound=h)
    if a>=1: out['reason']='linear_noncontractive'; return out
    disc=(1-a)**2-2*h*rn
    if disc<=0: out['reason']='no_majorant_root'; return out
    small=2*rn/(1-a+np.sqrt(disc))
    R=max(float(up(inflation*small)),1e-15)
    q=float(up(a+h*R))
    needed=float(up(rn+a*R+.5*h*R*R))
    if q>=1 or needed>R:
        out.update(reason='inclusion_failed',radius=R,contraction=q); return out
    E=min(R,float(up(rn/(1-q))))
    out.update(certified=True,radius=R,error=E,contraction=q,inclusion_slack=R-needed)
    return out

def paired_certificate(model_i,x_i,model_j,x_j,C=None,hessian_norms=None,inflation=1.05,max_steps=40):
    """Existence, uniqueness in paired box, and common/difference error bounds."""
    if model_i.n!=model_j.n: raise ValueError('The two state coordinate systems must agree.')
    if C is None: C=np.linalg.inv(model_i.jacobian(x_i))
    ri,rri,Ai,Ari=_preconditioned(model_i,x_i,C)
    rj,rrj,Aj,Arj=_preconditioned(model_j,x_j,C)
    rc,rrc=_linear_combine(ri,rri,rj,rrj,factor=.5)
    rd,rrd=_linear_combine(ri,rri,rj,rrj,sign=-1)
    am,amr=_linear_combine(Ai,Ari,Aj,Arj,factor=.5)
    ad,adr=_linear_combine(Ai,Ari,Aj,Arj,sign=-1)
    r=np.array([norm_inf(rc,rrc),norm_inf(rd,rrd)])
    a=norm_inf(am,amr); b=norm_inf(ad,adr)
    hp,hm=transformed_hessian_norms(C,model_i,model_j) if hessian_norms is None else hessian_norms
    hp=float(hp);hm=float(hm)
    out=dict(certified=False,residual_common=r[0],residual_difference=r[1],linear_common=a,linear_difference=b,hessian_sum=hp,hessian_difference=hm)
    if a+b/2>=1:
        out['reason']='linear_noncontractive';return out
    B0=np.array([[a,b/4],[b,a]])
    R=np.maximum(inflation*np.linalg.solve(np.eye(2)-B0,r),1e-15)
    for step in range(max_steps):
        c,d=R
        need=np.array([r[0]+a*c+b*d/4+hp*c*c/4+hm*c*d/4+hp*d*d/16,
                       r[1]+b*c+a*d+hm*c*c/2+hp*c*d/2+hm*d*d/8])
        need=up(need*(1+gamma(40)))
        aa=float(up(a+hp*c/2+hm*d/4));bb=float(up(b+hm*c+hp*d/2))
        q=float(up(aa+bb/2))
        if q>=1 or not np.all(np.isfinite(R)):
            out.update(reason='nonlinear_noncontractive',radius_common=c,radius_difference=d,contraction=q);return out
        if np.all(need<=R):
            break
        R=np.maximum(R,up(inflation*need))
    else:
        out.update(reason='inclusion_not_closed');return out
    # Inverse of positive two-by-two comparison matrix, with arithmetic guard.
    den=(1-aa)**2-bb*bb/4
    E=np.array([((1-aa)*r[0]+bb*r[1]/4)/den,(bb*r[0]+(1-aa)*r[1])/den])
    E=up(E*(1+gamma(50)/max(1-q,1e-12)))
    E=np.minimum(E,R)
    out.update(certified=True,radius_common=float(R[0]),radius_difference=float(R[1]),error_common=float(E[0]),error_difference=float(E[1]),contraction=q,inclusion_slack_common=float(R[0]-need[0]),inclusion_slack_difference=float(R[1]-need[1]),radius_iterations=step+1)
    return out

@dataclass
class QuadraticQuantity:
    constant: float
    linear: np.ndarray
    hessian: np.ndarray

    def __post_init__(self):
        self.linear=np.asarray(self.linear,dtype=float)
        self.hessian=self.hessian.toarray() if sparse.issparse(self.hessian) else np.asarray(self.hessian,dtype=float)

    def value_gradient_enclosure(self,x):
        qx,qr=dot_enclosure(self.hessian,np.asarray(x))
        gradient=self.linear+qx
        gr=qr+gamma(3)*(np.abs(self.linear)+np.abs(qx))
        value=self.constant+float(self.linear@x)+.5*float(np.asarray(x)@qx)
        scale=abs(self.constant)+float(np.abs(self.linear)@np.abs(x))+.5*float(np.abs(x)@(np.abs(self.hessian)@np.abs(x)))
        vr=gamma(4*len(x)+16)*scale
        return value,float(up(vr)),gradient,up(gr)

def quantity_interval(quantity,x,error):
    value,vr,p,pr=quantity.value_gradient_enclosure(x)
    eta=float(abs_sum(np.abs(p)+pr))*error+.5*float(abs_sum(quantity.hessian))*error*error+vr
    eta=float(up(eta*(1+gamma(20))))
    return dict(value=value,error=eta,lower=float(np.nextafter(value-eta,-np.inf)),upper=float(up(value+eta)))

def difference_interval(qi,xi,qj,xj,certificate):
    if not certificate['certified']: return dict(certified=False)
    vi,vir,pi,pir=qi.value_gradient_enclosure(xi)
    vj,vjr,pj,pjr=qj.value_gradient_enclosure(xj)
    pd,pdr=_linear_combine(pi,pir,pj,pjr,sign=-1)
    ps,psr=_linear_combine(pi,pir,pj,pjr)
    Qd=qi.hessian-qj.hessian; Qs=qi.hessian+qj.hessian
    Qr=gamma(3)*(np.abs(qi.hessian)+np.abs(qj.hessian))
    hd=float(abs_sum(np.abs(Qd)+Qr)); hs=float(abs_sum(np.abs(Qs)+Qr))
    c=certificate['error_common'];d=certificate['error_difference']
    eta=float(abs_sum(np.abs(pd)+pdr))*c+.5*float(abs_sum(np.abs(ps)+psr))*d+.5*hd*c*c+.5*hs*c*d+.125*hd*d*d
    eta+=vir+vjr+gamma(3)*(abs(vi)+abs(vj))
    eta=float(up(eta*(1+gamma(30))))
    val=vi-vj
    return dict(certified=True,value=val,error=eta,lower=float(np.nextafter(val-eta,-np.inf)),upper=float(up(val+eta)))

def recourse_difference_bounds(lower_matrix,upper_matrix,feasible_i,possible_i,feasible_j,possible_j):
    """For a separately CERTIFIED maximal service tier in each portfolio.

    F is a nonempty set of certified feasible controls. P contains at least one
    true minimum-cost control, and may contain unresolved controls. Initially
    F subset A subset P is sufficient; value-preserving domination pruning can
    subsequently remove nonoptimal controls from P, including feasible ones.
    Do not apply to arbitrary service tiers under a service-first controller.
    """
    L=np.asarray(lower_matrix);Uu=np.asarray(upper_matrix)
    fi=np.flatnonzero(feasible_i);pi=np.flatnonzero(possible_i)
    fj=np.flatnonzero(feasible_j);pj=np.flatnonzero(possible_j)
    if not len(fi) or not len(fj): raise ValueError('Each maximal service tier needs a certified feasible action.')
    if not len(pi) or not len(pj): raise ValueError('Each possible-optimal set must remain nonempty.')
    return float(np.min(np.max(L[np.ix_(pi,fj)],axis=1))),float(np.min(np.max(Uu[np.ix_(fi,pj)],axis=1)))
