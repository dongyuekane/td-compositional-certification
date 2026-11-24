"""Rectangular AC power flow and explicit quadratic coefficients.

All numerical experiments use balanced constant-PQ loads, fixed PV setpoints,
and explicit real transformer taps. No OPF or generator mode switching is hidden.
"""
from __future__ import annotations
import os,sys,re,time,itertools,hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'mpl_cache'))
VENDOR=Path(__file__).resolve().parents[1]/'vendor'
if VENDOR.exists():sys.path.append(str(VENDOR))
import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix,csr_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(1)
DATA=Path(__file__).parent/'data'

def case(name):
    if name=='case30':
        from pypower.case30 import case30

        source = case30()
        return {key: np.asarray(source[key]).copy() for key in ('bus', 'branch', 'gen', 'baseMVA')}
    text=(DATA/f'{name}.m').read_text();out={}
    for field in ['bus','branch','gen']:
        body=re.search(r'mpc\.'+field+r'\s*=\s*\[(.*?)\];',text,re.S).group(1)
        out[field]=np.array([[float(x) for x in l.split('%')[0].strip().rstrip(';').split()] for l in body.splitlines() if l.split('%')[0].strip()])
    out['baseMVA']=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',text).group(1))
    if name.startswith('case3') and name!='case33bw':return out
    if name not in ['case9','case30']:
        out['bus'][:,2:4]/=1000
        out['branch'][:,2:4]/=out['bus'][0,9]**2/out['baseMVA']
    return out

class ACSystem:
    def __init__(self,ppc,meta=None):
        self.ppc=ppc;self.meta=meta or {};self.base=float(ppc['baseMVA'])
        bus=ppc['bus'];br=ppc['branch'];gen=ppc['gen'];self.nb=len(bus)
        self.slack=int(np.where(bus[:,1]==3)[0][0]);self.ids=np.delete(np.arange(self.nb),self.slack)
        self.m=len(self.ids);self.n=2*self.m;self.pv=bus[self.ids,1]==2
        self.Y=np.diag((bus[:,4]+1j*bus[:,5])/self.base)
        self.Yf=np.zeros((len(br),self.nb),complex);self.Yt=self.Yf.copy()
        for k,e in enumerate(br):
            if e[10]==0:continue
            a,b=e[:2].astype(int)-1;y=1/complex(e[2],e[3]);sh=1j*e[4]/2
            t=(e[8] or 1.)*np.exp(1j*np.pi/180*e[9])
            self.Yf[k,a]=(y+sh)/abs(t)**2;self.Yf[k,b]=-y/np.conj(t)
            self.Yt[k,a]=-y/t;self.Yt[k,b]=y+sh
            self.Y[a]+=self.Yf[k];self.Y[b]+=self.Yt[k]
        self.G=self.Y.real;self.B=self.Y.imag
        pg=np.zeros(self.nb);qg=np.zeros(self.nb);self.vm=bus[:,7].copy()
        for g in gen:
            if g[7]:
                k=int(g[0])-1;pg[k]+=g[1];qg[k]+=g[2];self.vm[k]=g[5]
        self.spec=(pg-bus[:,2]+1j*(qg-bus[:,3]))/self.base
        self.vs=complex(self.vm[self.slack],0)
        self._hentries=None;self._lossH=None
    def voltage(self,x):
        v=np.full(self.nb,self.vs,complex);v[self.ids]=x[:self.m]+1j*x[self.m:];return v
    def pack(self,v):return np.r_[v[self.ids].real,v[self.ids].imag]
    def flat(self):return np.r_[self.vm[self.ids],np.zeros(self.m)]
    def residual(self,x):
        v=self.voltage(x);s=v*np.conj(self.Y@v);q=(s-self.spec).imag[self.ids]
        q[self.pv]=abs(v[self.ids][self.pv])**2-self.vm[self.ids][self.pv]**2
        return np.r_[(s-self.spec).real[self.ids],q]
    def jacobian(self,x):
        v=self.voltage(x);a=v.real;b=v.imag;ii=self.Y@v
        pa=a[:,None]*self.G+b[:,None]*self.B
        pb=-a[:,None]*self.B+b[:,None]*self.G
        qa=b[:,None]*self.G-a[:,None]*self.B
        qb=-b[:,None]*self.B-a[:,None]*self.G
        ix=np.arange(self.nb);pa[ix,ix]+=ii.real;pb[ix,ix]+=ii.imag
        qa[ix,ix]-=ii.imag;qb[ix,ix]+=ii.real
        ii=np.ix_(self.ids,self.ids)
        j=np.block([[pa[ii],pb[ii]],[qa[ii],qb[ii]]])
        for k in np.flatnonzero(self.pv):
            j[self.m+k]=0;j[self.m+k,k]=2*x[k];j[self.m+k,self.m+k]=2*x[self.m+k]
        return j
    def hessian_entries(self):
        if self._hentries is not None:return self._hentries
        entries={};mapping={int(v):i for i,v in enumerate(self.ids)};m=self.m
        def add(k,a,b,c):
            if c==0:return
            entries[k,a,b]=entries.get((k,a,b),0.)+c
            entries[k,b,a]=entries.get((k,b,a),0.)+c
        for k,nb in enumerate(self.ids):
            for ob in np.flatnonzero(abs(self.Y[nb])>0):
                if int(ob) not in mapping:continue
                l=mapping[int(ob)];g=self.G[nb,ob];b=self.B[nb,ob]
                add(k,k,l,g);add(k,m+k,m+l,g)
                add(k,m+k,l,b);add(k,k,m+l,-b)
                if not self.pv[k]:
                    add(m+k,m+k,l,g);add(m+k,k,m+l,-g)
                    add(m+k,k,l,-b);add(m+k,m+k,m+l,-b)
            if self.pv[k]:add(m+k,k,k,1.);add(m+k,m+k,m+k,1.)
        keys=sorted(k for k,v in entries.items() if v!=0.)
        self._hentries=tuple(np.array([k[d] for k in keys],int) for d in range(3))+(np.array([entries[k] for k in keys]),)
        return self._hentries
    @property
    def constant(self):return self.residual(np.zeros(self.n))
    @property
    def linear(self):return self.jacobian(np.zeros(self.n))
    def loss(self,x):
        v=self.voltage(x);return float(np.real(np.vdot(v,self.Y@v))*self.base)
    def loss_grad(self,x):
        v=self.voltage(x);gr=(self.Y+self.Y.conj().T)@v
        return self.base*np.r_[gr[self.ids].real,gr[self.ids].imag]
    def loss_hessian(self):
        if self._lossH is None:
            a=(self.Y+self.Y.conj().T)[np.ix_(self.ids,self.ids)]
            self._lossH=self.base*np.block([[a.real,-a.imag],[a.imag,a.real]])
        return self._lossH
    def solve(self,x0=None,tol=1e-10,maxiter=20,trace=False):
        x=self.flat() if x0 is None else np.array(x0,copy=True);hist=[x.copy()];start=time.perf_counter()
        for k in range(maxiter):
            f=self.residual(x)
            if np.max(abs(f))<=tol:break
            try:dx=solve(self.jacobian(x),-f,check_finite=False)
            except Exception:break
            x+=dx;hist.append(x.copy())
            if not np.all(np.isfinite(x)):break
        res=float(np.max(abs(self.residual(x))))
        return dict(x=x,voltage=self.voltage(x),converged=res<=tol,residual=res,iterations=len(hist)-1,trace=hist if trace else [],loss=self.loss(x),seconds=time.perf_counter()-start)
    def metrics(self,x):
        v=self.voltage(x);sf=v[self.ppc['branch'][:,0].astype(int)-1]*np.conj(self.Yf@v)*self.base
        st=v[self.ppc['branch'][:,1].astype(int)-1]*np.conj(self.Yt@v)*self.base
        rating=self.ppc['branch'][:,5];valid=rating>0
        thermal=float(np.max(np.maximum(abs(sf[valid]),abs(st[valid]))/rating[valid],initial=0.))
        injection=v*np.conj(self.Y@v)*self.base
        demand=self.ppc['bus'][:,2]+1j*self.ppc['bus'][:,3];generation=injection+demand
        gv=[]
        for g in self.ppc['gen']:
            z=generation[int(g[0])-1];gv.extend([(z.real-g[8])/max(1,g[8]),(g[9]-z.real)/max(1,g[8]),(z.imag-g[3])/max(1,abs(g[3])),(g[4]-z.imag)/max(1,abs(g[4]))])
        return dict(vmin=float(abs(v).min()),vmax=float(abs(v).max()),thermal=thermal,genviol=max(gv),loss=self.loss(x))

def radial_edges(fc,fault=-1,tie=-1):
    normal=fc['branch'][fc['branch'][:,10]>0].copy();ties=fc['branch'][fc['branch'][:,10]==0].copy()
    if fault>=0:
        normal=np.delete(normal,fault,axis=0)
        if tie>=0:
            e=ties[tie].copy();e[10]=1;normal=np.vstack([normal,e])
    adj=[[] for _ in fc['bus']]
    for e in normal:
        a,b=e[:2].astype(int)-1;adj[a].append(b);adj[b].append(a)
    seen={0};todo=[0]
    for a in todo:
        for b in adj[a]:
            if b not in seen:seen.add(b);todo.append(b)
    active=np.array(sorted(seen),int)
    normal=np.array([e for e in normal if int(e[0])-1 in seen and int(e[1])-1 in seen])
    return normal,active

def build_network(family='mixed9',load=1.,plan=None,control=None,event=0,shape_seed=0):
    """Two independently controlled physical feeders, without feeder replication.

    Transmission base power and all power ratings/demands/generation are scaled
    together from 100 to 10 MVA; transmission per-unit impedances are preserved.
    """
    plan=dict(T=0.,X=.5,F=1,S=1,R=1,C=1,**(plan or {})) if not plan else plan
    control=control or dict(taps=(1.075,1.075),caps=(1.2,1.2),service=1.,tie=-1)
    main=case('case30' if family=='mixed30' else 'case9');base=10.
    factor=base/float(main['baseMVA']);bus=main['bus'].copy();gen=main['gen'].copy();br=main['branch'].copy()
    bus[:,2:6]*=factor;gen[:,[1,2,3,4,8,9]]*=factor;br[:,5:8]*=factor
    bus[:,2:4]*=load;gen[:,1]*=load
    # Specify rated transfer capability for the two affected corridor branches.
    upgrade_ids=[2,4] if family!='mixed30' else [5,9]
    br[upgrade_ids,2:4]*=1-.3*plan['T'];br[upgrade_ids,5:8]*=1+.5*plan['T']
    if event in (1,3):br[2 if family!='mixed30' else 9,10]=0
    attach=[4,6] if family!='mixed30' else [7,20]
    names=['case33bw','case69'];feedmeta=[];requested=bus[:,2].sum();served=requested
    for k,(name,at) in enumerate(zip(names,attach)):
        fc=case(name);normal=fc['branch'][fc['branch'][:,10]>0]
        fault=9 if k==0 and event in (2,3) else -1
        tie=control.get('tie',-1) if fault>=0 else -1
        es,active=radial_edges(fc,fault,tie)
        rng=np.random.default_rng(shape_seed+1009*k)
        shape=np.clip(rng.lognormal(-.005,.07,len(fc['bus'])),.8,1.2) if shape_seed else np.ones(len(fc['bus']))
        nominal=fc['bus'][:,2:4].sum(axis=0)
        # Background demand is nonnegative, as in the submitted benchmark.
        old=bus[at,2:4].copy();bus[at,2:4]=np.maximum(old-nominal*load,0)
        requested+=float((fc['bus'][:,2]*load*shape).sum()-old[0]+bus[at,2])
        start=len(bus);fb=fc['bus'][active].copy();fb[:,0]=np.arange(start+1,start+len(fb)+1);fb[:,1]=1
        service=control.get('service',1.) if event else 1.
        fb[:,2:4]*=load*shape[active,None]*service
        fb[:,4:6]=0.;fb[:,7]=1.;fb[:,8]=0.
        loc={int(a):i for i,a in enumerate(active)}
        capnode=17 if k==0 else 60
        if capnode in loc:fb[loc[capnode],5]=control['caps'][k]
        bus=np.vstack([bus,fb]);tr_mva=5+2.5*plan['X'];tap=control['taps'][k]
        tr=np.array([[at+1,start+1,.006*base/tr_mva,.04*base/tr_mva,0,tr_mva,tr_mva,tr_mva,1/tap,0,1,-360,360]])
        zfactor=(1-.35*plan['F']) if k==0 else 1.
        ampsfactor=(1+.5*plan['F']) if k==0 else 1.
        eb=[]
        for j,e in enumerate(es):
            a,b=e[:2].astype(int)-1
            # Constant MVA limits, specified by nominal-voltage ampere equivalents.
            # These are not voltage-dependent current limits.
            row=np.zeros(13);row[:2]=[start+loc[a]+1,start+loc[b]+1]
            row[2:4]=e[2:4]*base/float(fc['baseMVA'])*zfactor
            amp=(400 if j<5 else 300)*ampsfactor
            row[5:8]=np.sqrt(3)*fc['bus'][0,9]*amp/1000
            row[10]=1;row[11:]=[-360,360];eb.append(row)
        tr_index=len(br);br=np.vstack([br,tr,np.array(eb)])
        feedmeta.append(dict(name=name,start=start,active=active.tolist(),tr_index=tr_index,source_bus=at,base_kv=float(fc['bus'][0,9]),capnode=start+loc[capnode] if capnode in loc else -1))
    actual=float(bus[:,2].sum());gen[:,1]*=actual/requested
    ppc=dict(version='2',baseMVA=base,bus=bus,gen=gen,branch=br)
    return ACSystem(ppc,dict(family=family,load=load,event=event,plan=dict(plan),control=dict(control),feeders=feedmeta,requested_mw=requested,served_mw=actual,unserved_mw=requested-actual))

if __name__=='__main__':
    from pypower.api import runpf,ppoption
    for family in ['mixed9','mixed30']:
        s=build_network(family);r=s.solve(trace=True);b,ok=runpf(s.ppc,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-11))
        v=b['bus'][:,7]*np.exp(1j*np.pi/180*b['bus'][:,8]);err=max(abs(v-r['voltage']))
        x=r['x'];rng=np.random.default_rng(1);d=rng.normal(0,.001,s.n)
        k,a,b,h=s.hessian_entries();quad=np.bincount(k,weights=.5*h*d[a]*d[b],minlength=s.n)
        identity=np.max(abs(s.residual(x+d)-s.residual(x)-s.jacobian(x)@d-quad))
        print(family,s.nb,r['converged'],r['iterations'],err,identity,s.metrics(x),flush=True)
        assert ok and err<1e-7 and identity<1e-10
