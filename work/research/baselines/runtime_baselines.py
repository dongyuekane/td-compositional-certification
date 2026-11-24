"""Strong exact-enumeration runtime baselines on the frozen normal dataset.

Only this directory is written. The original AC engine and benchmark are read
without modification. Sparse Newton, network reuse and warm starts change the
implementation, never the physical equations, controls or feasibility tests.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import sys,copy,itertools,json,time,hashlib

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
sys.path.insert(0,str(WORK/'simulation'))
sys.path.insert(0,str(WORK/'benchmarks'))
import ac
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import spsolve
from reference import catalog,LOSS_FACTOR

ORIGINAL_CASE=ac.case
CACHED_CASE=lru_cache(None)(ORIGINAL_CASE)
FAMILIES={"mixed9":[.9,1.1,1.25],"mixed30":[.6,.7,.8]}
SHAPE_SEED=20261108
TAPS=(1.04,1.075,1.1)
CAPS=(0.,1.2)


def snake(axes):
    if not axes:return [()]
    tail=snake(axes[1:]);result=[]
    for i,v in enumerate(axes[0]):
        result.extend((v,)+t for t in (tail if i%2==0 else tail[::-1]))
    return result


def state_key(plan,control):
    return (plan['T'],plan['X'],plan['F'],*control['taps'],*control['caps'])


class NetworkReuse:
    def __init__(self,family,load):
        self.family=family;self.load=load;self.templates={}
    def system(self,plan,control):
        key=(plan['T'],plan['X'],plan['F'])
        if key not in self.templates:
            basecontrol=dict(taps=(1.04,1.04),caps=(0.,0.),service=1.,tie=-1)
            self.templates[key]=ac.build_network(self.family,self.load,plan=plan,control=basecontrol,shape_seed=SHAPE_SEED)
        base=self.templates[key]
        s=copy.copy(base)
        s.ppc=dict(base.ppc,bus=base.ppc['bus'].copy(),branch=base.ppc['branch'].copy())
        s.Y=base.Y.copy();s.Yf=base.Yf.copy();s.Yt=base.Yt.copy()
        for k,meta in enumerate(base.meta['feeders']):
            row=meta['tr_index'];old=s.ppc['branch'][row].copy();a,b=old[:2].astype(int)-1
            s.Y[a]-=s.Yf[row];s.Y[b]-=s.Yt[row]
            s.ppc['branch'][row,8]=1/control['taps'][k]
            e=s.ppc['branch'][row];y=1/complex(e[2],e[3]);sh=1j*e[4]/2;t=e[8]*np.exp(1j*np.pi/180*e[9])
            s.Yf[row]=0.;s.Yt[row]=0.
            s.Yf[row,a]=(y+sh)/abs(t)**2;s.Yf[row,b]=-y/np.conj(t)
            s.Yt[row,a]=-y/t;s.Yt[row,b]=y+sh
            s.Y[a]+=s.Yf[row];s.Y[b]+=s.Yt[row]
            capnode=meta['capnode'];cap=control['caps'][k]
            if capnode>=0:
                s.ppc['bus'][capnode,5]=cap;s.Y[capnode,capnode]+=1j*cap/s.base
        s.G=s.Y.real;s.B=s.Y.imag;s._hentries=None;s._lossH=None
        s.meta=dict(base.meta,plan=dict(plan),control=dict(control))
        return s


class SparseNewton:
    def __init__(self,system):
        self.s=system;self.Y=csr_matrix(system.Y)
        sub=self.Y[system.ids,:][:,system.ids].tocoo()
        self.row=sub.row;self.col=sub.col;self.G=sub.data.real;self.B=sub.data.imag
        self.diagonal=self.row==self.col
        self.keepq=~system.pv[self.row]
        self.pvk=np.flatnonzero(system.pv)
    def residual(self,x):
        s=self.s;v=s.voltage(x);power=v*np.conj(self.Y@v)
        q=(power-s.spec).imag[s.ids]
        q[s.pv]=abs(v[s.ids][s.pv])**2-s.vm[s.ids][s.pv]**2
        return np.r_[(power-s.spec).real[s.ids],q]
    def jacobian(self,x):
        s=self.s;m=s.m;v=s.voltage(x);current=self.Y@v
        a=x[:m];b=x[m:];r=self.row;c=self.col;g=self.G;bb=self.B;diag=self.diagonal
        pa=a[r]*g+b[r]*bb;pb=-a[r]*bb+b[r]*g
        qa=b[r]*g-a[r]*bb;qb=-b[r]*bb-a[r]*g
        ir=current[s.ids].real;ii=current[s.ids].imag
        pa[diag]+=ir[r[diag]];pb[diag]+=ii[r[diag]]
        qa[diag]-=ii[r[diag]];qb[diag]+=ir[r[diag]]
        k=self.keepq;pv=self.pvk
        rows=np.r_[r,r,m+r[k],m+r[k],m+pv,m+pv]
        cols=np.r_[c,m+c,c[k],m+c[k],pv,m+pv]
        values=np.r_[pa,pb,qa[k],qb[k],2*a[pv],2*b[pv]]
        return coo_matrix((values,(rows,cols)),shape=(s.n,s.n)).tocsc()
    def solve(self,x0=None,tol=1e-10,maxiter=30):
        x=self.s.flat() if x0 is None else np.array(x0,copy=True)
        iterations=0;start=time.perf_counter()
        for _ in range(maxiter):
            f=self.residual(x)
            if max(abs(f))<=tol:break
            dx=spsolve(self.jacobian(x),-f);x+=dx;iterations+=1
            if not np.all(np.isfinite(x)):break
        residual=float(max(abs(self.residual(x))))
        return dict(x=x,converged=bool(residual<=tol),residual=residual,iterations=iterations,seconds=time.perf_counter()-start)


def verify_implementation():
    """Check independent assembly and sparse derivative identities before timing."""
    rows=[]
    ac.case=CACHED_CASE
    for family,load in [('mixed9',1.1),('mixed30',.8)]:
        maker=NetworkReuse(family,load)
        for T,X,F,ts,cs in [(0.,0.,0,(1.04,1.1),(0.,1.2)),(.5,.5,1,(1.075,1.04),(1.2,0.)),(1.,1.,1,(1.1,1.075),(1.2,1.2))]:
            p=dict(T=T,X=X,F=F,S=1,R=1,C=1);c=dict(taps=ts,caps=cs,service=1.,tie=-1)
            original=ac.build_network(family,load,plan=p,control=c,shape_seed=SHAPE_SEED)
            reuse=maker.system(p,c);sn=SparseNewton(reuse)
            x=reuse.flat()+np.random.default_rng(20261130).normal(0,.001,reuse.n)
            errors=dict(Y=float(np.max(abs(reuse.Y-original.Y))),Yf=float(np.max(abs(reuse.Yf-original.Yf))),Yt=float(np.max(abs(reuse.Yt-original.Yt))),residual=float(np.max(abs(sn.residual(x)-original.residual(x)))),jacobian=float(np.max(abs(sn.jacobian(x).toarray()-original.jacobian(x)))))
            assert max(errors.values())<2e-11,(family,errors)
            rows.append(dict(family=family,load=load,T=T,X=X,F=F,**errors))
    pd.DataFrame(rows).to_csv(HERE/'implementation_checks.csv',index=False)


def profile(family,load,method,previous_profile=None):
    tstart=time.perf_counter();cpu0=time.process_time()
    sparse=method.startswith('sparse');warm=method.endswith('warm');reuse=method!='original_dense_cold'
    ac.case=CACHED_CASE if reuse else ORIGINAL_CASE
    maker=NetworkReuse(family,load) if reuse else None
    grid=snake(((0.,.5,1.),(0.,.5,1.),(0,1)))
    controlgrid=snake((TAPS,TAPS,CAPS,CAPS))
    records=[];last=None;solutions={};fallbacks=0;warm_uses=0;cross_profile_uses=0
    for gi,(T,X,F) in enumerate(grid):
        p=dict(T=T,X=X,F=F,S=1,R=1,C=1)
        for ta,tb,ca,cb in (controlgrid if gi%2==0 else controlgrid[::-1]):
            c=dict(taps=(ta,tb),caps=(ca,cb),service=1.,tie=-1);key=state_key(p,c)
            assembled=time.perf_counter()
            s=maker.system(p,c) if reuse else ac.build_network(family,load,plan=p,control=c,shape_seed=SHAPE_SEED)
            solver=SparseNewton(s) if sparse else s
            assembly_seconds=time.perf_counter()-assembled
            x0=None;source='flat'
            if warm:
                starts=[('flat',s.flat())]
                if last is not None:starts.append(('adjacent_control_or_investment',last))
                if previous_profile is not None and key in previous_profile:starts.append(('previous_profile_same_state',previous_profile[key]))
                source,x0=min(starts,key=lambda item:float(np.max(abs(solver.residual(item[1])))))
                if source!='flat':warm_uses+=1
                if source=='previous_profile_same_state':cross_profile_uses+=1
            t0=time.perf_counter();r=solver.solve(x0=x0,tol=1e-10,maxiter=30);iterations=r['iterations']
            vm=abs(s.voltage(r['x']))
            if not r['converged'] or vm.min()<.5 or vm.max()>1.5:
                fallbacks+=1;r=solver.solve(x0=None,tol=1e-10,maxiter=30);iterations+=r['iterations'];source+=';flat_fallback'
            solve_seconds=time.perf_counter()-t0
            assert r['converged'],(family,load,key,method)
            last=r['x'].copy();solutions[key]=last
            m=s.metrics(last);margin=max((.9-m['vmin'])/.1,(m['vmax']-1.1)/.1,m['thermal']-1,m['genviol'])
            records.append(dict(family=family,load=load,method=method,T=T,X=X,F=F,tap0=ta,tap1=tb,cap0=ca,cap1=cb,iterations=iterations,initial_source=source,assembly_seconds=assembly_seconds,solve_seconds=solve_seconds,residual=r['residual'],feasible=margin<=1e-7,violation=margin,**m))
    elapsed=time.perf_counter()-tstart;cpu=time.process_time()-cpu0
    d=pd.DataFrame(records)
    old=pd.read_csv(WORK/f'benchmarks/pilot/unmodified/{family}_{load:.2f}_states.csv')
    keys=['T','X','F','tap0','tap1','cap0','cap1']
    joined=d.merge(old,on=keys,suffixes=('_new','_reference'),validate='one_to_one')
    assert len(joined)==648
    loss_error=float(abs(joined.loss_new-joined.loss_reference).max())
    voltage_error=float(max(abs(joined.vmin_new-joined.vmin_reference).max(),abs(joined.vmax_new-joined.vmax_reference).max()))
    disagreements=int((joined.feasible_new!=joined.feasible_reference).sum())
    assert loss_error<5e-8 and voltage_error<1e-8 and disagreements==0,(family,load,method,loss_error,voltage_error,disagreements)
    investments=[]
    for p in catalog():
        g=d[(d['T']==p['T'])&(d.X==p['X'])&(d.F==p['F'])]
        if not p['R']:g=g[(g.tap0==1.04)&(g.tap1==1.04)]
        if not p['C']:g=g[(g.cap0==0.)&(g.cap1==0.)]
        g=g[g.feasible]
        loss=float(g.loss.min()) if len(g) else np.nan
        investments.append(dict(family=family,load=load,method=method,plan=p['plan'],feasible=bool(len(g)),capital=p['capital'],loss=loss,objective=p['capital']+LOSS_FACTOR*loss))
    stat=dict(family=family,load=load,method=method,states=len(d),wall_seconds=elapsed,cpu_seconds=cpu,assembly_seconds=float(d.assembly_seconds.sum()),solve_seconds=float(d.solve_seconds.sum()),iterations=int(d.iterations.sum()),mean_iterations=float(d.iterations.mean()),warm_starts=warm_uses,previous_profile_starts=cross_profile_uses,flat_fallbacks=fallbacks,max_loss_error_mw=loss_error,max_voltage_error_pu=voltage_error,classification_disagreements=disagreements)
    return d,pd.DataFrame(investments),stat,solutions


def main():
    started=time.perf_counter();HERE.mkdir(parents=True,exist_ok=True)
    verify_implementation()
    methods=['original_dense_cold','dense_reuse_cold','sparse_reuse_cold','dense_reuse_warm','sparse_reuse_warm']
    protocol=dict(families=FAMILIES,shape_seed=SHAPE_SEED,states_per_profile=648,methods=methods,ordering='Serpentine adjacent T/X/F and independent tap/cap controls',warm_start='Choose smallest actual residual among flat, previous adjacent state and same control/investment from preceding load profile; no reference-optimum input',fallback='Cold flat solve if convergence or broad high-voltage-root checks fail',normal_full_supply=True,newton_tolerance=1e-10,capital_costs_source='benchmarks/reference.py',scope='Exact full control enumeration and shared normal investment per family; no screening and no oracle-guided state choice',ac_source_sha256=hashlib.sha256((WORK/'simulation/ac.py').read_bytes()).hexdigest())
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    stats=[];investment_tables=[]
    for method in methods:
        for family,loads in FAMILIES.items():
            previous=None
            for load in loads:
                d,iv,stat,previous=profile(family,load,method,previous if method.endswith('warm') else None)
                d.to_csv(HERE/f'{method}_{family}_{load:.2f}_states.csv',index=False)
                iv.to_csv(HERE/f'{method}_{family}_{load:.2f}_investments.csv',index=False)
                stats.append(stat);investment_tables.append(iv);print(json.dumps(stat),flush=True)
                (HERE/'profile_runtime.json').write_text(json.dumps(stats,indent=2)+'\n')
    iv=pd.concat(investment_tables,ignore_index=True);commons=[]
    for (method,family),g in iv.groupby(['method','family']):
        portfolios=g.groupby('plan').agg(feasible=('feasible','all'),objective=('objective','mean'))
        good=portfolios[portfolios.feasible];best=good.objective.idxmin()
        old=json.loads((WORK/'benchmarks/frozen_v1/normal_common_planning_summary.json').read_text())[family]
        assert int(best)==old['selected_plan'] and abs(float(good.loc[best,'objective'])-old['selected_objective'])<1e-7
        commons.append(dict(method=method,family=family,common_feasible_portfolios=len(good),selected_plan=int(best),objective=float(good.loc[best,'objective'])))
    sums=pd.DataFrame(stats).groupby('method').agg(states=('states','sum'),wall_seconds=('wall_seconds','sum'),cpu_seconds=('cpu_seconds','sum'),iterations=('iterations','sum'),assembly_seconds=('assembly_seconds','sum'),solve_seconds=('solve_seconds','sum'),warm_starts=('warm_starts','sum'),flat_fallbacks=('flat_fallbacks','sum')).reset_index()
    base=float(sums[sums.method=='original_dense_cold'].wall_seconds.iloc[0]);sums['speedup_vs_original']=base/sums.wall_seconds
    summary=dict(runtime=sums.to_dict('records'),common_investments=commons,all_reference_classifications_agree=True,max_loss_error_mw=max(s['max_loss_error_mw'] for s in stats),max_voltage_error_pu=max(s['max_voltage_error_pu'] for s in stats),interpretation='Timing is a single sequential run of each implementation, including model assembly, warm-start selection, solves and metrics; file output and reference checking are outside profile timings. Runtime optimization is a required baseline, not the claimed research novelty.',total_wall_seconds=time.perf_counter()-started)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
