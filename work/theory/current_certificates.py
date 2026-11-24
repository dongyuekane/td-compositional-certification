"""Equivalent current-injection AC certification with analytic regional bounds.

The power-flow simulator is unchanged. Its partial rectangular voltage state is
augmented by a PV reactive-power iterate and certified using an equivalent
current-balance equation. Large branch admittances are linear and therefore do
not appear in the nonlinear remainder. No sampled derivatives are used.
"""
import numpy as np
from certificates import (QuadraticModel,QuadraticQuantity,transformed_hessian_norms,
    dot_enclosure,norm_inf,_linear_combine,gamma,up)
from scipy import sparse

class CurrentModel:
    def __init__(self,system):
        self.system=system;self.m=system.m;self.pvid=np.flatnonzero(system.pv)
        self.n=system.n+len(self.pvid);self.nv=system.n
        self.pvpos={int(k):j for j,k in enumerate(self.pvid)}
        self.Y=system.Y[np.ix_(system.ids,system.ids)]
        self.y0=system.Y[system.ids,system.slack]*system.vs
        self.p=system.spec.real[system.ids];self.q=system.spec.imag[system.ids]
        self.linear=np.zeros((self.n,self.n))
        self.linear[:2*self.m,:2*self.m]=np.block([[self.Y.real,-self.Y.imag],[self.Y.imag,self.Y.real]])

    def augment(self,x):
        v=self.system.voltage(x)
        q=(v*np.conj(self.system.Y@v)).imag[self.system.ids][self.pvid]
        return np.r_[x,q]

    def parts(self,x):
        v=x[:self.m]+1j*x[self.m:2*self.m]
        q=self.q.copy();q[self.pvid]=x[2*self.m:]
        z=self.p-1j*q
        return v,z

    def residual(self,x):
        v,z=self.parts(x)
        f=self.Y@v+self.y0-z/np.conj(v)
        return np.r_[f.real,f.imag,abs(v[self.pvid])**2-self.system.vm[self.system.ids][self.pvid]**2]

    def jacobian(self,x):
        v,z=self.parts(x);w=np.conj(v)
        J=self.linear.copy();d=z/w**2
        ix=np.arange(self.m)
        J[ix,ix]+=d.real;J[ix,self.m+ix]+=(-1j*d).real
        J[self.m+ix,ix]+=d.imag;J[self.m+ix,self.m+ix]+=(-1j*d).imag
        for l,k in enumerate(self.pvid):
            col=2*self.m+l;t=1j/w[k]
            J[k,col]=t.real;J[self.m+k,col]=t.imag
            J[2*self.m+l,k]=2*v[k].real;J[2*self.m+l,self.m+k]=2*v[k].imag
        return J

    def enclosures(self,x):
        v,z=self.parts(x);w=np.conj(v)
        f=self.residual(x);J=self.jacobian(x)
        current_scale=np.abs(self.Y)@np.abs(v)+np.abs(self.y0)+np.abs(z)/np.abs(v)
        fscale=np.r_[current_scale,current_scale,abs(v[self.pvid])**2+self.system.vm[self.system.ids][self.pvid]**2]
        fr=gamma(6*self.m+32)*fscale
        js=np.abs(self.linear)
        ds=np.abs(z)/np.abs(w)**2;ix=np.arange(self.m)
        for ro,co in [(ix,ix),(ix,self.m+ix),(self.m+ix,ix),(self.m+ix,self.m+ix)]:js[ro,co]+=ds
        for l,k in enumerate(self.pvid):
            col=2*self.m+l;js[k,col]=js[self.m+k,col]=1/abs(w[k])
            js[2*self.m+l,k]=2*abs(v[k].real);js[2*self.m+l,self.m+k]=2*abs(v[k].imag)
        jr=gamma(64)*js
        return f,up(fr+1e-300),J,up(jr+1e-300)

    def hessian(self,x):
        v,z=self.parts(x);w=np.conj(v)
        rows=[];cols=[];values=[]
        def add_complex(k,a,b,value):
            for row,t in [(k,value.real),(self.m+k,value.imag)]:
                rows.append(row);cols.append(a*self.n+b);values.append(t)
        for k in range(self.m):
            a=k;b=self.m+k;t=-2*z[k]/w[k]**3
            add_complex(k,a,a,t);add_complex(k,a,b,-1j*t)
            add_complex(k,b,a,-1j*t);add_complex(k,b,b,-t)
            if k in self.pvpos:
                q=2*self.m+self.pvpos[k]
                for u,val in [(a,-1j/w[k]**2),(b,-1/w[k]**2)]:
                    add_complex(k,u,q,val);add_complex(k,q,u,val)
        for l,k in enumerate(self.pvid):
            for a in [k,self.m+k]:
                rows.append(2*self.m+l);cols.append(a*self.n+a);values.append(2.)
        return sparse.csr_matrix((values,(rows,cols)),shape=(self.n,self.n*self.n))

    def regional_hessian_variation(self,x,R):
        """Per residual-equation entrywise Hessian l1 change on ||dx||inf<=R.

        |Delta 1/conj(v)^p| <= p sqrt(2) R / vmin^(p+1).
        The PV reactive coefficient can change by at most R.
        """
        v,z=self.parts(x);vmin=np.abs(v)-np.sqrt(2)*R
        if np.min(vmin)<=0:return np.full(self.n,np.inf)
        qchange=np.zeros(self.m);qchange[self.pvid]=R
        dh=2*qchange/vmin**3+6*np.sqrt(2)*np.abs(z)*R/vmin**4
        dc=2*np.sqrt(2)*R/vmin**3
        bound=4*dh
        bound[self.pvid]+=4*dc[self.pvid]
        return up(np.r_[bound,bound,np.zeros(len(self.pvid))]*(1+gamma(50)))

    def center_hessian_roundoff(self,x,C):
        """Absolute input-Hessian guard, applied before possible cancellation."""
        v,z=self.parts(x)
        hscale=8*np.abs(z)/np.abs(v)**3
        hscale[self.pvid]+=4/np.abs(v[self.pvid])**2
        equation_scale=np.r_[hscale,hscale,np.zeros(len(self.pvid))]
        propagated=np.abs(C)@equation_scale
        return float(up(gamma(80)*max(propagated,default=0)*(1+gamma(self.n+8))))

    def loss_quantity(self):
        z=np.zeros(self.system.n)
        linear=np.zeros(self.n);linear[:self.nv]=self.system.loss_grad(z)
        H=np.zeros((self.n,self.n));H[:self.nv,:self.nv]=self.system.loss_hessian()
        return QuadraticQuantity(self.system.loss(z),linear,H)

def current_preconditioned(model,x,C):
    f,fr,J,Jr=model.enclosures(x)
    r,rr=dot_enclosure(-C,f,fr);cj,cjr=dot_enclosure(C,J,Jr)
    A=np.eye(model.n)-cj;Ar=cjr+gamma(3)*(np.eye(model.n)+np.abs(cj))
    return r,rr,A,up(Ar)

def center_hessian_model(model,x):
    return QuadraticModel(np.zeros(model.n),np.zeros((model.n,model.n)),model.hessian(x))

def transformed_variation(C,model,x,R):
    v=model.regional_hessian_variation(x,R)
    if not np.all(np.isfinite(v)):return np.inf
    vv=np.abs(C)@v
    return float(up(max(vv,default=0)*(1+gamma(model.n+8))))

def current_single_certificate(model,x,C=None,inflation=1.05,max_steps=40):
    if C is None:C=np.linalg.inv(model.jacobian(x))
    r,rr,A,Ar=current_preconditioned(model,x,C)
    rn=norm_inf(r,rr);a=norm_inf(A,Ar)
    H0=transformed_hessian_norms(C,center_hessian_model(model,x))
    # Account for evaluation of the local analytic Hessian at the center.
    H0=float(up(H0+model.center_hessian_roundoff(x,C)))
    out=dict(certified=False,residual_bound=rn,linear_bound=a,center_hessian_bound=H0)
    if a>=1:out['reason']='linear_noncontractive';return out
    R=max(float(up(inflation*rn/(1-a))),1e-15)
    for k in range(max_steps):
        h=H0+transformed_variation(C,model,x,R)
        q=float(up(a+h*R));need=float(up((rn+a*R+.5*h*R*R)*(1+gamma(20))))
        if q>=1 or not np.isfinite(q):
            out.update(reason='nonlinear_noncontractive',radius=R,contraction=q,hessian_bound=h);return out
        if need<=R:break
        R=float(up(inflation*need))
    else:out['reason']='inclusion_not_closed';return out
    E=min(R,float(up(rn/(1-q))))
    out.update(certified=True,radius=R,error=E,contraction=q,hessian_bound=h,inclusion_slack=R-need,radius_iterations=k+1)
    return out

def current_paired_certificate(mi,xi,mj,xj,Ci=None,Cj=None,inflation=1.05,max_steps=40):
    if mi.n!=mj.n:raise ValueError('Paired coordinates and PV sets must agree.')
    if Ci is None:Ci=np.linalg.inv(mi.jacobian(xi))
    if Cj is None:Cj=np.linalg.inv(mj.jacobian(xj))
    ri,rri,Ai,Ari=current_preconditioned(mi,xi,Ci)
    rj,rrj,Aj,Arj=current_preconditioned(mj,xj,Cj)
    rc,rrc=_linear_combine(ri,rri,rj,rrj,factor=.5)
    rd,rrd=_linear_combine(ri,rri,rj,rrj,sign=-1)
    am,amr=_linear_combine(Ai,Ari,Aj,Arj,factor=.5)
    ad,adr=_linear_combine(Ai,Ari,Aj,Arj,sign=-1)
    r=np.array([norm_inf(rc,rrc),norm_inf(rd,rrd)])
    a=norm_inf(am,amr);b=norm_inf(ad,adr)
    hi=center_hessian_model(mi,xi);hj=center_hessian_model(mj,xj)
    hp0,hm0=transformed_hessian_norms(Ci,hi,hj,C_j=Cj)
    absolute_hround=mi.center_hessian_roundoff(xi,Ci)+mj.center_hessian_roundoff(xj,Cj)
    hp0=float(up(hp0+absolute_hround));hm0=float(up(hm0+absolute_hround))
    out=dict(certified=False,residual_common=r[0],residual_difference=r[1],linear_common=a,linear_difference=b,center_hessian_sum=hp0,center_hessian_difference=hm0)
    if a+b/2>=1:out['reason']='linear_noncontractive';return out
    B0=np.array([[a,b/4],[b,a]])
    R=np.maximum(inflation*np.linalg.solve(np.eye(2)-B0,r),1e-15)
    for k in range(max_steps):
        c,d=R;physical_R=c+d/2
        variation=transformed_variation(Ci,mi,xi,physical_R)+transformed_variation(Cj,mj,xj,physical_R)
        hp=hp0+variation;hm=hm0+variation
        need=np.array([r[0]+a*c+b*d/4+hp*c*c/4+hm*c*d/4+hp*d*d/16,
                       r[1]+b*c+a*d+hm*c*c/2+hp*c*d/2+hm*d*d/8])
        need=up(need*(1+gamma(40)))
        aa=float(up(a+hp*c/2+hm*d/4));bb=float(up(b+hm*c+hp*d/2))
        q=float(up(aa+bb/2))
        if q>=1 or not np.isfinite(q):
            out.update(reason='nonlinear_noncontractive',radius_common=c,radius_difference=d,contraction=q,hessian_sum=hp,hessian_difference=hm);return out
        if np.all(need<=R):break
        R=np.maximum(R,up(inflation*need))
    else:out['reason']='inclusion_not_closed';return out
    den=(1-aa)**2-bb*bb/4
    E=np.array([((1-aa)*r[0]+bb*r[1]/4)/den,(bb*r[0]+(1-aa)*r[1])/den])
    E=up(E*(1+gamma(50)/max(1-q,1e-12)));E=np.minimum(E,R)
    out.update(certified=True,radius_common=float(R[0]),radius_difference=float(R[1]),error_common=float(E[0]),error_difference=float(E[1]),contraction=q,hessian_sum=hp,hessian_difference=hm,inclusion_slack_common=float(R[0]-need[0]),inclusion_slack_difference=float(R[1]-need[1]),radius_iterations=k+1)
    return out

def _complex_product_guard(A,b):
    A=np.asarray(A);b=np.asarray(b)
    value=A@b
    radius=gamma(8*A.shape[-1]+24)*(np.abs(A)@np.abs(b))
    return value,up(radius+1e-300)

def electrical_enclosures(model,x,certificate,use_network_structure=True):
    """Enclose voltage, terminal current and generated powers at certified root.

    Optional current-balance substitution avoids multiplying arbitrary unrelated
    neighboring voltage errors by very large feeder admittances. The approximate
    inverse defect is retained, so no exact numerical matrix inverse is assumed.
    """
    if not certificate.get('certified'):return dict(certified=False)
    s=model.system;E=certificate['error'];v=s.voltage(x[:model.nv])
    dv=np.full(s.nb,np.sqrt(2)*E);dv[s.slack]=0.
    current_f=s.Yf@v;current_t=s.Yt@v;current_bus=s.Y@v
    rv=gamma(8*s.nb+24)*(np.abs(s.Y)@np.abs(v))
    rf=gamma(8*s.nb+24)*(np.abs(s.Yf)@np.abs(v))
    rt=gamma(8*s.nb+24)*(np.abs(s.Yt)@np.abs(v))
    if not use_network_structure:
        ifr=np.abs(s.Yf)@dv+rf;itr=np.abs(s.Yt)@dv+rt;ibr=np.abs(s.Y)@dv+rv
        vc=v.copy();vr=dv.copy()
    else:
        smallv,z=model.parts(x);vmin=np.abs(smallv)-dv[s.ids]
        if min(vmin)<=0:return dict(certified=False,reason='voltage_disk_contains_zero')
        dq=np.zeros(model.m);dq[model.pvid]=E
        injrad=np.abs(z)*dv[s.ids]/(np.abs(smallv)*vmin)+dq/vmin
        injrad=up(injrad*(1+gamma(30)))
        f,fr,_,_=model.enclosures(x);res=f[:model.m]+1j*f[model.m:2*model.m]
        resrad=np.hypot(fr[:model.m],fr[model.m:2*model.m])
        Z=np.linalg.inv(model.Y)
        def transform(A,initial,initial_rad):
            K,Kr=_complex_product_guard(A,Z)
            correction,cr=_complex_product_guard(K,res)
            defect,dr=_complex_product_guard(K,model.Y)
            defect=A-defect
            dr+=gamma(4)*(np.abs(A)+np.abs(K@model.Y))
            radius=np.abs(K)@(injrad+resrad)+Kr@np.abs(res)
            radius+=(np.abs(defect)+dr)@dv[s.ids]
            radius+=initial_rad+cr
            return initial-correction,up(radius*(1+gamma(4*model.m+30)))
        vc=v.copy();vr=dv.copy()
        newv,newvr=transform(np.eye(model.m),v[s.ids],np.zeros(model.m))
        vc[s.ids]=newv;vr[s.ids]=newvr
        current_f,ifr=transform(s.Yf[:,s.ids],current_f,rf)
        current_t,itr=transform(s.Yt[:,s.ids],current_t,rt)
        current_bus,ibr=transform(s.Y[:,s.ids],current_bus,rv)
    def product_power(busidx,I,Ir):
        vv=vc[busidx];vv_r=vr[busidx]
        val=s.base*vv*np.conj(I)
        rad=s.base*(np.abs(vv)*Ir+np.abs(I)*vv_r+vv_r*Ir)
        rad+=gamma(12)*np.abs(val)
        return val,up(rad*(1+gamma(20)))
    br=s.ppc['branch'];frombus=br[:,0].astype(int)-1;tobus=br[:,1].astype(int)-1
    sf,sfr=product_power(frombus,current_f,ifr);st,strad=product_power(tobus,current_t,itr)
    sb,sbr=product_power(np.arange(s.nb),current_bus,ibr)
    gen=sb+s.ppc['bus'][:,2]+1j*s.ppc['bus'][:,3]
    pglo=gen.real-sbr;pghi=gen.real+sbr;qglo=gen.imag-sbr;qghi=gen.imag+sbr
    # Non-slack specified P is exact; PV Q is explicitly in the augmented state.
    for local,bus in enumerate(s.ids):
        pglo[bus]=pghi[bus]=model.p[local]*s.base+s.ppc['bus'][bus,2]
        if local in model.pvpos:
            qcenter=x[2*model.m+model.pvpos[local]]*s.base+s.ppc['bus'][bus,3]
            qglo[bus]=qcenter-E*s.base;qghi[bus]=qcenter+E*s.base
        else:
            qglo[bus]=qghi[bus]=model.q[local]*s.base+s.ppc['bus'][bus,3]
    return dict(certified=True,voltage_lower=np.maximum(0,np.abs(vc)-vr),voltage_upper=np.abs(vc)+vr,
                flow_from_lower=np.maximum(0,np.abs(sf)-sfr),flow_from_upper=np.abs(sf)+sfr,
                flow_to_lower=np.maximum(0,np.abs(st)-strad),flow_to_upper=np.abs(st)+strad,
                generation_p_lower=pglo,generation_p_upper=pghi,generation_q_lower=qglo,generation_q_upper=qghi,
                voltage_center=vc,voltage_radius=vr,current_from_center=current_f,current_from_radius=ifr)

def feasibility_certificate(model,x,certificate,vmin=.9,vmax=1.1,tolerance=1e-7,use_network_structure=True):
    enc=electrical_enclosures(model,x,certificate,use_network_structure)
    if not enc.get('certified'):return dict(status='unresolved',reason=enc.get('reason','no_state_certificate'))
    s=model.system;lo=[];hi=[];labels=[]
    def add(label,l,h):
        labels.append(label);lo.append(float(l));hi.append(float(h))
    for k,(a,b) in enumerate(zip(enc['voltage_lower'],enc['voltage_upper'])):
        add('voltage_low_'+str(k),(vmin-b)/.1,(vmin-a)/.1)
        add('voltage_high_'+str(k),(a-vmax)/.1,(b-vmax)/.1)
    for k,e in enumerate(s.ppc['branch']):
        if e[10]<=0 or e[5]<=0:continue
        a=max(enc['flow_from_lower'][k],enc['flow_to_lower'][k])/e[5]-1
        b=max(enc['flow_from_upper'][k],enc['flow_to_upper'][k])/e[5]-1
        add('branch_mva_'+str(k),a,b)
    for j,g in enumerate(s.ppc['gen']):
        if not g[7]:continue
        k=int(g[0])-1;pl=enc['generation_p_lower'][k];pu=enc['generation_p_upper'][k]
        ql=enc['generation_q_lower'][k];qu=enc['generation_q_upper'][k]
        add('generator_p_upper_'+str(j),(pl-g[8])/max(1,g[8]),(pu-g[8])/max(1,g[8]))
        add('generator_p_lower_'+str(j),(g[9]-pu)/max(1,g[8]),(g[9]-pl)/max(1,g[8]))
        add('generator_q_upper_'+str(j),(ql-g[3])/max(1,abs(g[3])),(qu-g[3])/max(1,abs(g[3])))
        add('generator_q_lower_'+str(j),(g[4]-qu)/max(1,abs(g[4])),(g[4]-ql)/max(1,abs(g[4])))
    lower=max(lo);upper=max(hi)
    status='feasible' if upper<=tolerance else 'violating' if lower>tolerance else 'unresolved'
    return dict(status=status,violation_lower=lower,violation_upper=upper,
                decisive_lower=labels[int(np.argmax(lo))],decisive_upper=labels[int(np.argmax(hi))],
                semantics='Constraint status at the uniquely certified local AC operating root; not a global AC infeasibility claim.')

def service_tier_certificate(service,statuses):
    """Apply existence quantifiers to a finite service-first control menu."""
    service=np.asarray(service,dtype=float);statuses=np.asarray(statuses)
    feasible=statuses=='feasible';possible=statuses!='violating'
    if not np.any(possible):return dict(status='no_admissible_control_at_certified_roots')
    if not np.any(feasible):return dict(status='unresolved')
    lower=float(max(service[feasible]));upper=float(max(service[possible]))
    if lower!=upper:return dict(status='unresolved',service_lower=lower,service_upper=upper)
    return dict(status='certified',service=lower,feasible=feasible&(service==lower),possible=possible&(service==lower))
