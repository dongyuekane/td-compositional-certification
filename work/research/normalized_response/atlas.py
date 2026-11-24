"""Exploratory voltage/load-normalized component responses. Not yet certified.

For a fixed feeder Y and load shape d, v=u*w(lambda/u**2). Each discrete
asset/control label retains its own response. No convexification of controls.
The first prototype checks cost/accuracy before building an error certificate.
"""
from pathlib import Path
import sys,time,itertools,json
import numpy as np
from numpy.polynomial import chebyshev as ch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'benchmarks'))
import ac, reference
from scipy.linalg import solve
from scipy.interpolate import PPoly
from math import comb
from functools import lru_cache

@lru_cache(None)
def power_conversion(degree,width):
    basis=np.zeros((degree+1,degree+1))
    for k in range(degree+1):
        c=np.zeros(degree+1);c[k]=1
        q=ch.cheb2poly(c);basis[:len(q),k]=q
    shift=np.zeros_like(basis);scale=2/width
    for j in range(degree+1):
        for k in range(j,degree+1):shift[j,k]=scale**j*comb(k,j)*(-1)**(k-j)
    return shift@basis

SEED=20261108

def extract_feeder(system,k):
    meta=system.meta['feeders'][k]
    inds=np.r_[meta['source_bus'],np.arange(meta['start'],meta['start']+len(meta['active']))]
    ppc=system.ppc
    bus=ppc['bus'][inds].copy();bus[:,0]=np.arange(1,len(inds)+1)
    bus[0,1]=3;bus[0,2:6]=0;bus[0,7]=1;bus[0,8]=0
    mapping={int(g+1):i+1 for i,g in enumerate(inds)}
    rows=np.array([int(e[0]) in mapping and int(e[1]) in mapping for e in ppc['branch']])
    br=ppc['branch'][rows].copy()
    br[:,:2]=[[mapping[int(a)],mapping[int(b)]] for a,b in br[:,:2]]
    gen=np.zeros((1,ppc['gen'].shape[1]));gen[0,[0,3,4,5,6,7,8,9]]=[1,100,-100,1,10,1,100,0]
    return ac.ACSystem(dict(baseMVA=ppc['baseMVA'],bus=bus,branch=br,gen=gen),dict(feeder=k))

def feeder_at(k,x,f,tap,cap):
    p=dict(T=0.,X=x,F=f if k==0 else 0,S=0,R=1,C=1)
    c=dict(taps=(tap,tap),caps=(cap,cap),service=1.,tie=-1)
    return extract_feeder(ac.build_network('mixed9',1.,p,c,shape_seed=SEED),k)

class Response:
    def __init__(self,feeder,degree=12,domain=(.4,1.8)):
        start=time.perf_counter();self.system=feeder;self.domain=domain
        nodes=np.cos(np.pi*np.arange(degree+1)/degree)
        zs=.5*(sum(domain)+(domain[1]-domain[0])*nodes)
        original=feeder.spec.copy();values=[];iters=0;x0=None
        for z in zs[::-1]:
            feeder.spec=original*z
            r=feeder.solve(x0=x0,maxiter=30)
            if not r['converged']:raise ValueError(f'Feeder failed at normalized load {z}')
            x0=r['x'];iters+=r['iterations']
            v=r['voltage'];power=v[0]*np.conj(feeder.Y[0]@v)
            values.append(np.r_[power.real,power.imag,v[1:].real,v[1:].imag])
        feeder.spec=original
        self.coeff=ch.chebfit(nodes,np.array(values[::-1]),degree)
        self.dcoeff=ch.chebder(self.coeff,axis=0)*2/(domain[1]-domain[0])
        shifted=power_conversion(degree,domain[1]-domain[0])@self.coeff
        self.polynomial=PPoly(shifted[::-1,None,:],np.array(domain))
        self.derivative=self.polynomial.derivative()
        self.iterations=iters;self.samples=degree+1;self.seconds=time.perf_counter()-start
    def evaluate(self,z):
        return self.polynomial(z),self.derivative(z)
    def voltage(self,source,load):
        y,_=self.evaluate(load/abs(source)**2);m=self.system.m
        return source*(y[2:2+m]+1j*y[2+m:])

class Atlas:
    def __init__(self,degree=12,domain=(.4,1.8)):
        self.responses={};start=time.perf_counter()
        for k in (0,1):
            for x,f,tap,cap in itertools.product((0.,.5,1.),(0,1) if k==0 else (0,),reference.TAPS,reference.CAPS):
                self.responses[k,x,f,tap,cap]=Response(feeder_at(k,x,f,tap,cap),degree,domain)
        self.seconds=time.perf_counter()-start
    def get(self,k,plan,control):
        return self.responses[k,plan['X'],plan['F'] if k==0 else 0,control['taps'][k],control['caps'][k]]

class Reduced:
    def __init__(self,system,atlas):
        self.full=system;self.atlas=atlas;self.load=system.meta['load']
        n=system.meta['feeders'][0]['start'];p=system.ppc
        ppc=dict(baseMVA=p['baseMVA'],bus=p['bus'][:n].copy(),gen=p['gen'].copy(),branch=p['branch'][(p['branch'][:,0]<=n)&(p['branch'][:,1]<=n)].copy())
        self.main=ac.ACSystem(ppc)
        self.responses=[atlas.get(k,system.meta['plan'],system.meta['control']) for k in (0,1)]
        self.sources=[fm['source_bus'] for fm in system.meta['feeders']]
    def residual_jac(self,x,jac=True):
        s=self.main;v=s.voltage(x);r=s.residual(x);j=s.jacobian(x) if jac else None
        for source,response in zip(self.sources,self.responses):
            idx=int(np.where(s.ids==source)[0][0]);u2=abs(v[source])**2;z=self.load/u2
            y,dy=response.evaluate(z)
            r[idx]+=u2*y[0]
            if not s.pv[idx]:r[s.m+idx]+=u2*y[1]
            if jac:
                a=2*(y[:2]-z*dy[:2]);gr=np.array([v[source].real,v[source].imag])
                j[idx,[idx,s.m+idx]]+=a[0]*gr
                if not s.pv[idx]:j[s.m+idx,[idx,s.m+idx]]+=a[1]*gr
        return r,j
    def solve(self,x0=None):
        start=time.perf_counter();s=self.main;x=s.flat() if x0 is None else x0.copy()
        for it in range(20):
            f,j=self.residual_jac(x)
            if np.max(abs(f))<1e-11:break
            x+=solve(j,-f,check_finite=False)
        v=np.zeros(self.full.nb,complex);v[:s.nb]=s.voltage(x)
        for k,(source,response) in enumerate(zip(self.sources,self.responses)):
            fm=self.full.meta['feeders'][k];startidx=fm['start']
            v[startidx:startidx+response.system.m]=response.voltage(v[source],self.load)
        fullx=self.full.pack(v)
        return dict(x=fullx,main_x=x,voltage=v,iterations=it,residual=float(np.max(abs(self.full.residual(fullx)))),loss=self.full.loss(fullx),seconds=time.perf_counter()-start)

def pilot():
    out=[]
    for degree in (6,10,14):
        atlas=Atlas(degree);start=time.perf_counter();maxv=maxloss=maxres=0.;disagree=0;rows=[]
        for family,load in [('mixed9',1.25),('mixed30',.8)]:
            for t,x,f in itertools.product((0.,1.),(0.,1.),(0,1)):
                p=dict(T=t,X=x,F=f,S=0,R=1,C=1)
                for c in list(reference.controls(p))[::7]:
                    s=reference.build(family,load,p,c,shape_seed=SEED)
                    reduced=Reduced(s,atlas).solve();truth=s.solve()
                    err=max(abs(truth['voltage']-reduced['voltage']))
                    maxv=max(maxv,err);maxloss=max(maxloss,abs(truth['loss']-reduced['loss']));maxres=max(maxres,reduced['residual'])
                    rows.append(dict(family=family,load=load,T=t,X=x,F=f,**c,error=float(err),residual=reduced['residual'],seconds=reduced['seconds']))
        record=dict(degree=degree,training_seconds=atlas.seconds,training_pf=sum(r.samples for r in atlas.responses.values()),training_newton=sum(r.iterations for r in atlas.responses.values()),test_states=len(rows),max_voltage_error=maxv,max_loss_error=maxloss,max_residual=maxres,mean_reduced_seconds=np.mean([r['seconds'] for r in rows]))
        out.append(record);print(json.dumps(record),flush=True)
    (Path(__file__).parent/'pilot.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':pilot()
