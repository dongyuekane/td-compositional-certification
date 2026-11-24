"""Exploratory end-to-end response-atlas timing against the strong exact baseline.

Each family pays for its own 54-response training. All 3,888 existing normal
states are evaluated; no action or investment is removed by reference labels.
This validates numerical agreement only and is not a certified surrogate.
"""
from pathlib import Path
import sys,time,json,hashlib,itertools

HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'baselines'))
from atlas import Atlas
import runtime_baselines as baseline
from runtime_baselines import ac,np,pd,NetworkReuse,SparseNewton,snake,state_key,FAMILIES,SHAPE_SEED,TAPS,CAPS
from scipy.linalg import solve
from reference import catalog,LOSS_FACTOR

OUT=HERE/'study_results';OUT.mkdir(exist_ok=True)


class MainTemplates:
    def __init__(self):self.templates={}
    def get(self,full):
        key=full.meta['plan']['T']
        if key not in self.templates:
            n=full.meta['feeders'][0]['start'];p=full.ppc
            keep=(p['branch'][:,0]<=n)&(p['branch'][:,1]<=n)
            ppc=dict(baseMVA=p['baseMVA'],bus=p['bus'][:n].copy(),gen=p['gen'].copy(),branch=p['branch'][keep].copy())
            main=ac.ACSystem(ppc);sources=[m['source_bus'] for m in full.meta['feeders']]
            self.templates[key]=(main,sources,[int(np.flatnonzero(main.ids==s)[0]) for s in sources])
        return self.templates[key]


class FastReduced:
    def __init__(self,full,atlas,main_templates):
        self.full=full;self.load=full.meta['load']
        self.main,self.sources,self.sourceids=main_templates.get(full)
        self.responses=[atlas.get(k,full.meta['plan'],full.meta['control']) for k in (0,1)]
        self.zmin=np.inf;self.zmax=-np.inf;self.domain_violation=False
    def residual_jac(self,x,jac=True):
        s=self.main;v=s.voltage(x);r=s.residual(x);j=s.jacobian(x) if jac else None
        for source,idx,response in zip(self.sources,self.sourceids,self.responses):
            u2=abs(v[source])**2;z=self.load/u2
            self.zmin=min(self.zmin,z);self.zmax=max(self.zmax,z)
            if not response.domain[0]<=z<=response.domain[1]:self.domain_violation=True
            y,dy=response.evaluate(z);r[idx]+=u2*y[0]
            if not s.pv[idx]:r[s.m+idx]+=u2*y[1]
            if jac:
                a=2*(y[:2]-z*dy[:2]);gradient=np.array([v[source].real,v[source].imag])
                j[idx,[idx,s.m+idx]]+=a[0]*gradient
                if not s.pv[idx]:j[s.m+idx,[idx,s.m+idx]]+=a[1]*gradient
        return r,j
    def residual(self,x):return self.residual_jac(x,False)[0]
    def solve(self,x0=None,tol=1e-11,maxiter=25):
        s=self.main;x=s.flat() if x0 is None else np.array(x0,copy=True);iterations=0
        for _ in range(maxiter):
            f,j=self.residual_jac(x)
            if max(abs(f))<=tol:break
            x+=solve(j,-f,check_finite=False);iterations+=1
        reduced_residual=float(max(abs(self.residual(x))))
        v=np.zeros(self.full.nb,complex);v[:s.nb]=s.voltage(x)
        for k,(source,response) in enumerate(zip(self.sources,self.responses)):
            start=self.full.meta['feeders'][k]['start'];v[start:start+response.system.m]=response.voltage(v[source],self.load)
        fullx=self.full.pack(v)
        return dict(x=fullx,main_x=x,voltage=v,iterations=iterations,converged=bool(reduced_residual<=tol),reduced_residual=reduced_residual,full_residual=float(max(abs(self.full.residual(fullx)))),zmin=self.zmin,zmax=self.zmax,domain_violation=self.domain_violation)


def investments(d,family,load,method):
    out=[]
    for p in catalog():
        g=d[(d['T']==p['T'])&(d.X==p['X'])&(d.F==p['F'])]
        if not p['R']:g=g[(g.tap0==1.04)&(g.tap1==1.04)]
        if not p['C']:g=g[(g.cap0==0.)&(g.cap1==0.)]
        g=g[g.feasible]
        loss=float(g.loss.min()) if len(g) else np.nan
        out.append(dict(family=family,load=load,method=method,plan=p['plan'],feasible=bool(len(g)),objective=p['capital']+LOSS_FACTOR*loss,capital=p['capital'],loss=loss))
    return pd.DataFrame(out)


def atlas_profile(family,load,atlas,degree,previous_profile=None):
    method=f'atlas_degree{degree}';started=time.perf_counter();cpu0=time.process_time()
    maker=NetworkReuse(family,load);main_templates=MainTemplates();last=None;solutions={};rows=[];xs=[]
    physical_grid=snake(((0.,.5,1.),(0.,.5,1.),(0,1)));control_grid=snake((TAPS,TAPS,CAPS,CAPS))
    for gi,(T,X,F) in enumerate(physical_grid):
        p=dict(T=T,X=X,F=F,S=1,R=1,C=1)
        for ta,tb,ca,cb in (control_grid if gi%2==0 else control_grid[::-1]):
            c=dict(taps=(ta,tb),caps=(ca,cb),service=1.,tie=-1);key=state_key(p,c)
            t0=time.perf_counter();s=maker.system(p,c);reduced=FastReduced(s,atlas,main_templates);assembly_seconds=time.perf_counter()-t0
            starts=[('flat',reduced.main.flat())]
            if last is not None:starts.append(('adjacent_control_or_investment',last))
            if previous_profile is not None and key in previous_profile:starts.append(('previous_profile_same_state',previous_profile[key]))
            source,x0=min(starts,key=lambda a:float(max(abs(reduced.residual(a[1])))))
            t0=time.perf_counter();r=reduced.solve(x0);iterations=r['iterations'];fallback=False
            if not r['converged'] or r['domain_violation']:
                # An out-of-domain map may never silently provide a planning
                # classification. A genuine exact fallback is charged here.
                fallback=True;exact=SparseNewton(s).solve(tol=1e-10)
                assert exact['converged'];r['x']=exact['x'];r['full_residual']=exact['residual'];iterations+=exact['iterations']
            solve_seconds=time.perf_counter()-t0
            last=r['main_x'].copy();solutions[key]=last
            m=s.metrics(r['x']);margin=max((.9-m['vmin'])/.1,(m['vmax']-1.1)/.1,m['thermal']-1,m['genviol'])
            rows.append(dict(family=family,load=load,method=method,T=T,X=X,F=F,tap0=ta,tap1=tb,cap0=ca,cap1=cb,loss=m['loss'],vmin=m['vmin'],vmax=m['vmax'],thermal=m['thermal'],genviol=m['genviol'],violation=margin,feasible=bool(margin<=1e-7),iterations=iterations,initial_source=source,assembly_seconds=assembly_seconds,solve_seconds=solve_seconds,full_residual=r['full_residual'],reduced_residual=r['reduced_residual'],zmin=r['zmin'],zmax=r['zmax'],exact_fallback=fallback))
            xs.append(r['x'])
    wall=time.perf_counter()-started;cpu=time.process_time()-cpu0
    d=pd.DataFrame(rows)
    old=pd.read_csv(WORK/f'benchmarks/pilot/unmodified/{family}_{load:.2f}_states.csv').reset_index(names='reference_index')
    traces=np.load(WORK/f'benchmarks/pilot/unmodified/{family}_{load:.2f}_traces.npz')['x']
    keys=['T','X','F','tap0','tap1','cap0','cap1'];joined=d.merge(old,on=keys,suffixes=('_new','_reference'),validate='one_to_one',sort=False)
    assert len(joined)==648
    # Cartesian coordinate differences equal complex-voltage differences up to
    # their Euclidean combination; evaluate complex errors for a direct report.
    voltage_error=0.
    for i,a in enumerate(joined.itertuples()):
        n=len(xs[i]);m=n//2;error=xs[i]-traces[int(a.reference_index),-1,:n]
        voltage_error=max(voltage_error,float(np.max(np.hypot(error[:m],error[m:]))))
    joined['loss_error_mw']=joined.loss_new-joined.loss_reference
    joined['classification_agrees']=joined.feasible_new==joined.feasible_reference
    stat=dict(family=family,load=load,method=method,degree=degree,states=len(d),wall_seconds=wall,cpu_seconds=cpu,assembly_seconds=float(d.assembly_seconds.sum()),solve_seconds=float(d.solve_seconds.sum()),iterations=int(d.iterations.sum()),exact_fallbacks=int(d.exact_fallback.sum()),classification_disagreements=int((~joined.classification_agrees).sum()),false_feasible=int((joined.feasible_new&~joined.feasible_reference).sum()),false_infeasible=int((~joined.feasible_new&joined.feasible_reference).sum()),max_loss_error_mw=float(joined.loss_error_mw.abs().max()),max_complex_voltage_error_pu=voltage_error,max_full_residual=float(d.full_residual.max()),normalized_load_min=float(d.zmin.min()),normalized_load_max=float(d.zmax.max()),main_templates=len(main_templates.templates))
    return d,investments(d,family,load,method),stat,solutions,joined


def common_result(iv,family,method):
    portfolios=iv.groupby('plan').agg(feasible=('feasible','all'),objective=('objective','mean'))
    good=portfolios[portfolios.feasible]
    if not len(good):return dict(family=family,method=method,common_feasible_portfolios=0,selected_plan=None)
    pid=int(good.objective.idxmin());value=float(good.loc[pid,'objective'])
    reference=json.loads((WORK/'benchmarks/frozen_v1/normal_common_planning_summary.json').read_text())[family]
    truth=pd.read_csv(WORK/f'benchmarks/frozen_v1/{family}_normal_common_portfolios.csv').set_index('plan')
    return dict(family=family,method=method,common_feasible_portfolios=len(good),selected_plan=pid,approximate_objective=value,reference_selected_plan=reference['selected_plan'],selected_plan_matches=pid==reference['selected_plan'],selected_reference_feasible=bool(truth.loc[pid,'feasible']),selected_reference_objective=float(truth.loc[pid,'objective']),true_objective_regret=float(truth.loc[pid,'objective']-reference['selected_objective']),objective_error_at_selected=float(value-truth.loc[pid,'objective']))


def main():
    started=time.perf_counter();sourcehash=hashlib.sha256((HERE/'atlas.py').read_bytes()).hexdigest()
    ac.case=baseline.CACHED_CASE
    protocol=dict(status='Exploratory approximation, not a rigorous certificate',families=FAMILIES,shape_seed=SHAPE_SEED,degrees=[6,10],normalized_domain=[.4,1.8],labels_per_family=54,training_cost_policy='An independently constructed Atlas is charged in full to every family and degree; no cross-family response sharing or free offline training',normal_states_per_profile=648,reduced_newton_tolerance=1e-11,baseline='Network reuse plus sparse Newton with adjacent-control/investment and prior-profile residual-selected warm starts, tolerance 1e-10',main_reuse='One main-grid ACSystem for each T level in each profile; all X/F/tap/cap combinations preserve their discrete feeder response label',timing_scope='Training plus online state enumeration including assembly, warm-start selection, solves and physical metrics. File I/O, reference comparisons and common post-processing of investment menus are outside the comparative timing.',atlas_source_sha256=sourcehash)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    runtimes=[];profile_records=[];planning=[]
    for family,loads in FAMILIES.items():
        previous=None;ivs=[];stats=[]
        for load in loads:
            d,iv,stat,previous=baseline.profile(family,load,'sparse_reuse_warm',previous)
            profile_records.append(stat);stats.append(stat);ivs.append(iv)
        method='sparse_reuse_warm';planning.append(common_result(pd.concat(ivs),family,method))
        runtimes.append(dict(family=family,method=method,degree=None,training_seconds=0.,training_samples=0,training_newton_updates=0,online_seconds=sum(s['wall_seconds'] for s in stats),total_seconds=sum(s['wall_seconds'] for s in stats),online_newton_updates=sum(s['iterations'] for s in stats),states=sum(s['states'] for s in stats),classification_disagreements=0))
        print('BASELINE',json.dumps(runtimes[-1]),flush=True)
    for degree in [6,10]:
        for family,loads in FAMILIES.items():
            ac.case=baseline.CACHED_CASE;t0=time.perf_counter();atlas=Atlas(degree=degree,domain=(.4,1.8));train=time.perf_counter()-t0
            previous=None;stats=[];ivs=[]
            for load in loads:
                d,iv,stat,previous,joined=atlas_profile(family,load,atlas,degree,previous)
                profile_records.append(stat);stats.append(stat);ivs.append(iv)
                d.to_csv(OUT/f'degree{degree}_{family}_{load:.2f}_states.csv',index=False)
                joined.to_csv(OUT/f'degree{degree}_{family}_{load:.2f}_reference_comparison.csv',index=False)
                print('ATLAS PROFILE',json.dumps(stat),flush=True)
            method=f'atlas_degree{degree}';planning.append(common_result(pd.concat(ivs),family,method))
            runtime=dict(family=family,method=method,degree=degree,training_seconds=train,training_samples=sum(r.samples for r in atlas.responses.values()),training_newton_updates=sum(r.iterations for r in atlas.responses.values()),online_seconds=sum(s['wall_seconds'] for s in stats),total_seconds=train+sum(s['wall_seconds'] for s in stats),online_newton_updates=sum(s['iterations'] for s in stats),states=sum(s['states'] for s in stats),classification_disagreements=sum(s['classification_disagreements'] for s in stats),max_full_residual=max(s['max_full_residual'] for s in stats),max_loss_error_mw=max(s['max_loss_error_mw'] for s in stats),max_complex_voltage_error_pu=max(s['max_complex_voltage_error_pu'] for s in stats),exact_fallbacks=sum(s['exact_fallbacks'] for s in stats))
            base=next(r['total_seconds'] for r in runtimes if r['family']==family and r['method']=='sparse_reuse_warm')
            runtime.update(speedup_including_training=base/runtime['total_seconds'],time_reduction_including_training=1-runtime['total_seconds']/base)
            runtimes.append(runtime);print('ATLAS FAMILY',json.dumps(runtime),flush=True)
            (OUT/'profile_runtime.json').write_text(json.dumps(profile_records,indent=2)+'\n')
    totals=[]
    for method in ['sparse_reuse_warm','atlas_degree6','atlas_degree10']:
        rr=[r for r in runtimes if r['method']==method]
        totals.append(dict(method=method,families=len(rr),states=sum(r['states'] for r in rr),training_seconds=sum(r['training_seconds'] for r in rr),online_seconds=sum(r['online_seconds'] for r in rr),total_seconds=sum(r['total_seconds'] for r in rr),classification_disagreements=sum(r['classification_disagreements'] for r in rr),training_samples=sum(r['training_samples'] for r in rr)))
    for r in totals:r['speedup_including_training']=totals[0]['total_seconds']/r['total_seconds'];r['time_reduction_including_training']=1-r['total_seconds']/totals[0]['total_seconds']
    assert hashlib.sha256((HERE/'atlas.py').read_bytes()).hexdigest()==sourcehash,'Atlas source changed during run; rerun a stable version.'
    result=dict(protocol=protocol,family_runtime=runtimes,totals=totals,common_planning=planning,total_audit_wall_seconds=time.perf_counter()-started)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
