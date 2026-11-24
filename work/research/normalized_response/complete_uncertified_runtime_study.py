"""Matched complete-planning times for two uncertified contextual baselines.

State enumeration, point-valued action-menu minimization, and common portfolio
selection are all timed. Reference audit and file I/O are outside the timers.
The two methods share exactly the same menu/portfolio aggregation code. Each
response run independently trains 54 degree-six labels for each family.
"""
from pathlib import Path
import hashlib, json, time
from run_study import (Atlas, baseline, ac, np, pd, NetworkReuse, MainTemplates,
    FastReduced, SparseNewton, snake, state_key, FAMILIES, SHAPE_SEED, TAPS,
    CAPS, catalog, LOSS_FACTOR, WORK, investments)

HERE=Path(__file__).resolve().parent
OUT=HERE/'complete_uncertified_runtime_results'
KEYS=['T','X','F','tap0','tap1','cap0','cap1']
NONREDUNDANT={p['plan'] for p in catalog() if p['S']==0}


def profile(family,load,method,atlas=None,previous=None):
    tstart=time.perf_counter();cpu0=time.process_time()
    maker=NetworkReuse(family,load);templates=MainTemplates()
    physical=snake(((0.,.5,1.),(0.,.5,1.),(0,1)))
    controls=snake((TAPS,TAPS,CAPS,CAPS))
    rows=[];last=None;solutions={};predictions=[];fallbacks=0
    for gi,(T,X,F) in enumerate(physical):
        p=dict(T=T,X=X,F=F,S=1,R=1,C=1)
        for ta,tb,ca,cb in (controls if gi%2==0 else controls[::-1]):
            c=dict(taps=(ta,tb),caps=(ca,cb),service=1.,tie=-1)
            key=state_key(p,c);s=maker.system(p,c)
            solver=SparseNewton(s) if method=='sparse_reuse_warm' else FastReduced(s,atlas,templates)
            initial=s.flat() if method=='sparse_reuse_warm' else solver.main.flat()
            starts=[('flat',initial)]
            if last is not None:starts.append(('adjacent_control_or_investment',last))
            if previous is not None and key in previous:starts.append(('previous_profile_same_state',previous[key]))
            source,x0=min(starts,key=lambda a:float(max(abs(solver.residual(a[1])))))
            if method=='sparse_reuse_warm':
                r=solver.solve(x0=x0,tol=1e-10,maxiter=30);iterations=r['iterations']
                vm=abs(s.voltage(r['x']))
                if not r['converged'] or vm.min()<.5 or vm.max()>1.5:
                    fallbacks+=1;r=solver.solve(tol=1e-10,maxiter=30);iterations+=r['iterations']
                assert r['converged']
                last=r['x'].copy();residual=r['residual']
            else:
                r=solver.solve(x0);iterations=r['iterations']
                if not r['converged'] or r['domain_violation']:
                    fallbacks+=1;exact=SparseNewton(s).solve(tol=1e-10)
                    assert exact['converged'];r['x']=exact['x'];iterations+=exact['iterations']
                    r['full_residual']=exact['residual']
                last=r['main_x'].copy();residual=r['full_residual']
            solutions[key]=last;predictions.append(r['x'].copy())
            m=s.metrics(r['x']);violation=max((.9-m['vmin'])/.1,(m['vmax']-1.1)/.1,m['thermal']-1,m['genviol'])
            rows.append(dict(family=family,load=load,method=method,T=T,X=X,F=F,
                       tap0=ta,tap1=tb,cap0=ca,cap1=cb,iterations=iterations,
                       feasible=bool(violation<=1e-7),violation=violation,
                       residual=residual,initial_source=source,**m))
    online=time.perf_counter()-tstart;cpu=time.process_time()-cpu0
    d=pd.DataFrame(rows)
    # Both uncertified methods use the identical point-valued minimization.
    t=time.perf_counter();menu=investments(d,family,load,method);menu_time=time.perf_counter()-t
    stat=dict(family=family,load=load,method=method,states=len(d),
              online_seconds=online,online_cpu_seconds=cpu,
              menu_aggregation_seconds=menu_time,iterations=int(d.iterations.sum()),
              fallbacks=fallbacks,prediction_sha256=hashlib.sha256(np.asarray(predictions).tobytes()).hexdigest())
    # Audit begins only after all measured state/menu work has finished.
    truth=pd.read_csv(WORK/f'benchmarks/pilot/unmodified/{family}_{load:.2f}_states.csv')
    joined=d.merge(truth,on=KEYS,suffixes=('','_reference'),validate='one_to_one')
    stat['classification_disagreements']=int((joined.feasible!=joined.feasible_reference).sum())
    stat['max_loss_error_mw']=float((joined.loss-joined.loss_reference).abs().max())
    assert len(joined)==648 and stat['classification_disagreements']==0,stat
    if method=='sparse_reuse_warm':assert stat['max_loss_error_mw']<5e-8,stat
    return menu,stat,solutions


def select_common(menus):
    portfolios=menus.groupby('plan').agg(feasible=('feasible','all'),objective=('objective','mean'))
    feasible=portfolios[portfolios.feasible]
    if not len(feasible):return portfolios,dict(selected_plan=None,common_feasible_portfolios=0)
    pid=int(feasible.objective.idxmin())
    return portfolios,dict(selected_plan=pid,selected_objective=float(feasible.loc[pid,'objective']),
                           common_feasible_portfolios=len(feasible))


def main():
    OUT.mkdir(exist_ok=True);ac.case=baseline.CACHED_CASE
    paths=[HERE/'atlas.py',HERE/'run_study.py',HERE.parent/'baselines/runtime_baselines.py']
    hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(repetitions=3,families=FAMILIES,shape_seed=SHAPE_SEED,response_degree=6,
       normalized_domain=[.4,1.8],labels_per_family=54,states_per_profile=648,
       methods=['sparse_reuse_warm','response_uncertified'],
       timing_scope='Independent response training for each family/run, all state assembly/initialization/solution/physical evaluation, 144 menu aggregations per profile, and common investment selection. Reference file I/O/audit and output I/O are excluded. Both methods use identical point-menu and common-selection functions.',
       repeated_order='Method order alternates by repetition plus family index. No cross-run solution or trained-response reuse. Static case parsing cache is shared by all methods.',
       catalog='The code aggregates the same original 144 entries as the certified timing pipeline. S=1 is analytically dominated in normal operation; the resulting 72 nonredundant entries are checked after timed selection.',
       comparison='Contextual uncertified runtime only; the claimed validation speedup compares two independently certified pipelines.',source_sha256=hashes)
    profiles=[];families=[];planning=[]
    for rep in range(3):
        for fi,(family,loads) in enumerate(FAMILIES.items()):
            methods=['sparse_reuse_warm','response_uncertified']
            if (rep+fi)%2:methods=methods[::-1]
            for method in methods:
                atlas=None;train=0.
                if method=='response_uncertified':
                    t=time.perf_counter();atlas=Atlas(degree=6,domain=(.4,1.8));train=time.perf_counter()-t
                previous=None;menus=[];stats=[]
                for load in loads:
                    menu,stat,previous=profile(family,load,method,atlas,previous)
                    stat['repetition']=rep+1;profiles.append(stat);stats.append(stat);menus.append(menu)
                t=time.perf_counter();portfolio,result=select_common(pd.concat(menus));selection=time.perf_counter()-t
                # The nonredundant analytic catalog audit is outside the timer.
                p72=portfolio[portfolio.index.isin(NONREDUNDANT)];f72=p72[p72.feasible]
                result.update(repetition=rep+1,family=family,method=method,catalog72_feasible=len(f72),
                              catalog72_selected_plan=int(f72.objective.idxmin()))
                truth=json.loads((WORK/'benchmarks/frozen_v1/normal_common_planning_summary.json').read_text())[family]
                result['reference_selected_plan']=truth['selected_plan']
                result['selected_plan_matches']=result['selected_plan']==truth['selected_plan']
                result['objective_error_at_selected']=result['selected_objective']-truth['selected_objective']
                assert result['selected_plan_matches'] and result['catalog72_selected_plan']==result['selected_plan'],result
                planning.append(result)
                row=dict(repetition=rep+1,family=family,method=method,states=sum(s['states'] for s in stats),
                    training_seconds=train,training_samples=0 if atlas is None else sum(r.samples for r in atlas.responses.values()),
                    online_seconds=sum(s['online_seconds'] for s in stats),
                    menu_aggregation_seconds=sum(s['menu_aggregation_seconds'] for s in stats),
                    common_selection_seconds=selection,classification_disagreements=sum(s['classification_disagreements'] for s in stats))
                row['total_seconds']=sum(row[k] for k in ['training_seconds','online_seconds','menu_aggregation_seconds','common_selection_seconds'])
                families.append(row);print('FAMILY',json.dumps(row),flush=True)
    repeated=[]
    for rep in range(1,4):
        for method in ['sparse_reuse_warm','response_uncertified']:
            rr=[r for r in families if r['repetition']==rep and r['method']==method]
            row=dict(repetition=rep,method=method,families=len(rr))
            for k in ['states','training_seconds','training_samples','online_seconds','menu_aggregation_seconds','common_selection_seconds','total_seconds','classification_disagreements']:
                row[k]=sum(r[k] for r in rr)
            repeated.append(row)
    summary=[]
    for method in ['sparse_reuse_warm','response_uncertified']:
        rr=[r for r in repeated if r['method']==method]
        totals=[r['total_seconds'] for r in rr]
        row=dict(method=method,repetitions=3,states_per_run=3888,
                 total_seconds_by_repetition=totals,median_total_seconds=float(np.median(totals)),
                 min_total_seconds=min(totals),max_total_seconds=max(totals))
        for key in ['training_seconds','online_seconds','menu_aggregation_seconds','common_selection_seconds']:
            row['median_'+key]=float(np.median([r[key] for r in rr]))
        summary.append(row)
    for row in summary:
        row['median_speedup_vs_sparse_uncertified']=summary[0]['median_total_seconds']/row['median_total_seconds']
        row['median_time_reduction_vs_sparse_uncertified']=1-row['median_total_seconds']/summary[0]['median_total_seconds']
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(WORK))],'Source changed.'
    result=dict(protocol=protocol,summary=summary,repeated_totals=repeated,family_runtime=families,
                profile_statistics=profiles,common_planning=planning)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
