"""Independent end-to-end certified-planning pipelines on all 3,888 states.

Both pipelines build their own degree-six response atlas per network family,
use identical cache and residual-selected warm-start policies, and certify the
same predicted states. The full-network baseline applies residual correction
and first tries its cheaper direct electrical-output enclosure. The structural
current-output enclosure is used only if the direct bound is unresolved.
Reference data are read after the profile timer and never guide construction.
"""
from pathlib import Path
import sys, time, json, hashlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from full_certificate_ablation import (Atlas,CurrentModel,current_single_certificate,
    current_residual_enclosure,dot_enclosure,corrected_state_enclosure,
    corrected_loss_interval,feasibility_certificate,compositional_certificate,
    certify_response,source_hashes,np,pd,WORK)
from run_study import (baseline,ac,NetworkReuse,MainTemplates,FastReduced,snake,
    state_key,FAMILIES,TAPS,CAPS,SHAPE_SEED,LOSS_FACTOR)
from certify_study import menu_bounds

OUT=HERE/'full_certificate_planning_results'
KEYS=['T','X','F','tap0','tap1','cap0','cap1']


def full_certificate(system,predicted):
    started=time.perf_counter()
    model=CurrentModel(system);x=model.augment(predicted['x'])
    C=np.linalg.inv(model.jacobian(x))
    cert=current_single_certificate(model,x,C)
    if not cert['certified']:
        return dict(certified=False,status='unresolved',seconds=time.perf_counter()-started,
                    loss_lower=-np.inf,loss_upper=np.inf,dimension=model.n,
                    reason=cert.get('reason'),structured_output_fallback=False)
    f,fr=current_residual_enclosure(model,x);r,rr=dot_enclosure(-C,f,fr)
    corrected=corrected_state_enclosure(model,x,C,cert,r,rr)
    loss=corrected_loss_interval(model,x,C,cert,correction=r,correction_radius=rr)
    outputcert=dict(certified=True,error=corrected['error'])
    status=feasibility_certificate(model,corrected['center'],outputcert,
                                   use_network_structure=False)
    structured=status['status']=='unresolved'
    if structured:
        status=feasibility_certificate(model,corrected['center'],outputcert,
                                       use_network_structure=True)
    return dict(certified=True,status=status['status'],seconds=time.perf_counter()-started,
                loss_lower=loss['lower'],loss_upper=loss['upper'],dimension=model.n,
                state_error=cert['error'],corrected_state_error=corrected['error'],
                violation_lower=status['violation_lower'],violation_upper=status['violation_upper'],
                structured_output_fallback=structured)


def profile(family,load,atlas,tubes,method,previous=None):
    start=time.perf_counter();cpu0=time.process_time()
    maker=NetworkReuse(family,load);templates=MainTemplates()
    last=None;solutions={};records=[];predictions=[]
    physical=snake(((0.,.5,1.),(0.,.5,1.),(0,1)))
    controls=snake((TAPS,TAPS,CAPS,CAPS))
    for gi,(T,X,F) in enumerate(physical):
        plan=dict(T=T,X=X,F=F,S=1,R=1,C=1)
        for ta,tb,ca,cb in (controls if gi%2==0 else controls[::-1]):
            control=dict(taps=(ta,tb),caps=(ca,cb),service=1.,tie=-1)
            key=state_key(plan,control)
            t=time.perf_counter();system=maker.system(plan,control)
            reduced=FastReduced(system,atlas,templates);assembly=time.perf_counter()-t
            t=time.perf_counter();starts=[('flat',reduced.main.flat())]
            if last is not None:starts.append(('adjacent_control_or_investment',last))
            if previous is not None and key in previous:starts.append(('previous_profile_same_state',previous[key]))
            source,x0=min(starts,key=lambda a:float(max(abs(reduced.residual(a[1])))))
            warm=time.perf_counter()-t
            t=time.perf_counter();predicted=reduced.solve(x0);solve=time.perf_counter()-t
            assert predicted['converged'] and not predicted['domain_violation'],(family,load,key)
            if method=='full_current_corrected':
                result=full_certificate(system,predicted)
            else:
                localtubes=[tubes[k,X,F if k==0 else 0,control['taps'][k],control['caps'][k]] for k in (0,1)]
                result=compositional_certificate(reduced,predicted,localtubes)
            last=predicted['main_x'].copy();solutions[key]=last
            predictions.append(predicted['x'].copy())
            row=dict(family=family,load=load,degree=6,method=method,T=T,X=X,F=F,
                     tap0=ta,tap1=tb,cap0=ca,cap1=cb,iterations=predicted['iterations'],
                     initial_source=source,assembly_seconds=assembly,warm_selection_seconds=warm,
                     solve_seconds=solve,full_residual=predicted['full_residual'],
                     reduced_residual=predicted['reduced_residual'],**result)
            row['loss_width']=row['loss_upper']-row['loss_lower'];records.append(row)
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu0
    frame=pd.DataFrame(records)
    # All reference I/O and comparisons occur after the measured pipeline.
    truth=pd.read_csv(WORK/f'benchmarks/pilot/unmodified/{family}_{load:.2f}_states.csv')
    joined=frame.merge(truth,on=KEYS,suffixes=('','_reference'),validate='one_to_one',sort=False)
    assert len(joined)==648
    joined['loss_covered']=(joined.loss>=joined.loss_lower-1e-9)&(joined.loss<=joined.loss_upper+1e-9)
    joined['violation_covered']=(joined.violation>=joined.violation_lower-1e-8)&(joined.violation<=joined.violation_upper+1e-8)
    joined['decision_correct']=((joined.status=='feasible')&joined.feasible)|((joined.status=='violating')&~joined.feasible)|(joined.status=='unresolved')
    summary=dict(family=family,load=load,method=method,states=len(frame),wall_seconds=wall,cpu_seconds=cpu,
                 certified=int(frame.certified.sum()),status_counts=frame.status.value_counts().to_dict(),
                 classification_errors=int((~joined.decision_correct).sum()),
                 loss_coverage_failures=int((~joined.loss_covered).sum()),
                 violation_coverage_failures=int((~joined.violation_covered).sum()),
                 prediction_sha256=hashlib.sha256(np.asarray(predictions).tobytes()).hexdigest(),
                 max_loss_width_mw=float(frame.loss_width.max()),median_loss_width_mw=float(frame.loss_width.median()),
                 max_dimension=int(frame.dimension.max()),iterations=int(frame.iterations.sum()),
                 structured_output_fallbacks=int(frame.get('structured_output_fallback',pd.Series(dtype=bool)).sum()))
    for name in ['assembly_seconds','warm_selection_seconds','solve_seconds','seconds']:
        summary['certificate_seconds' if name=='seconds' else name]=float(frame[name].sum())
    assert summary['classification_errors']==summary['loss_coverage_failures']==summary['violation_coverage_failures']==0,summary
    return frame,summary,solutions,joined


def aggregate_common(menus):
    rows=[]
    for plan,g in menus.groupby('plan'):
        assert len(g)==3
        status='violating' if (g.status=='violating').any() else 'feasible' if (g.status=='feasible').all() else 'unresolved'
        rows.append(dict(plan=int(plan),status=status,
                         lower=float(g.capital.iloc[0]+LOSS_FACTOR*g.loss_lower.mean()),
                         upper=float(g.capital.iloc[0]+LOSS_FACTOR*g.loss_upper.mean())))
    frame=pd.DataFrame(rows);feasible=frame[frame.status=='feasible']
    if not len(feasible):return frame,dict(selected_plan=None,certified_optimal=False)
    winner=feasible.loc[feasible.upper.idxmin()]
    rivals=frame[(frame.plan!=winner.plan)&(frame.status!='violating')]
    runner=rivals.loc[rivals.lower.idxmin()];gap=float(runner.lower-winner.upper)
    result=dict(selected_plan=int(winner.plan),selected_lower=float(winner.lower),
                selected_upper=float(winner.upper),nearest_rival_plan=int(runner.plan),
                certified_gap_million_cny=gap,certified_optimal=bool(gap>0),
                common_status_counts=frame.status.value_counts().to_dict())
    return frame,result


def main():
    OUT.mkdir(parents=True,exist_ok=True);hashes=source_hashes()
    hashes['research/normalized_response/certify_study.py']=hashlib.sha256((HERE/'certify_study.py').read_bytes()).hexdigest()
    ac.case=baseline.CACHED_CASE
    protocol=dict(families=FAMILIES,shape_seed=SHAPE_SEED,degree=6,domain=[.4,1.8],states_per_profile=648,
       pipelines=['full_current_corrected','compositional'],
       timing='Independent complete pipelines; each family/method independently builds its 54-response atlas. Online includes assembly, warm-start choice, reduced solve, fresh certificate/preconditioner, loss/constraint bounds, and result collection. Offline tube certificates are charged only to composition. Menu aggregation and common-investment selection are included in total. Reference I/O and validation are outside timers.',
       baseline='Full-network current-injection inclusion with fresh Jacobian inverse; residual-corrected state and quadratic loss; direct electrical bounds first, structural current-output fallback only when unresolved.',
       same_prediction='SHA256 of every profile prediction array must match between independently rerun pipelines.',
       fallback='No certificate failure silently determines a planning result. Unresolved action remains possible in menu lower bound. No reference-dependent fallback.',
       scope='Certified finite-menu planning evaluation at the locally certified high-voltage root; guarded binary64 implementation of real-arithmetic theorems, not formal interval arithmetic.',
       source_sha256=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    profiles=[];families=[];planning=[]
    for fi,(family,loads) in enumerate(FAMILIES.items()):
        methods=['full_current_corrected','compositional']
        if fi%2:methods=methods[::-1]
        for method in methods:
            t=time.perf_counter();atlas=Atlas(degree=6,domain=(.4,1.8));train=time.perf_counter()-t
            tubes={};offline=0.
            if method=='compositional':
                t=time.perf_counter();tubes={key:certify_response(r) for key,r in atlas.responses.items()};offline=time.perf_counter()-t
                assert all(t['certified'] for t in tubes.values())
            previous=None;ss=[];menus=[];menu_seconds=0.
            for load in loads:
                frame,stat,previous,joined=profile(family,load,atlas,tubes,method,previous)
                t=time.perf_counter();menu=menu_bounds(frame);aggregation=time.perf_counter()-t
                menu_seconds+=aggregation;stat['menu_seconds']=aggregation
                ss.append(stat);profiles.append(stat);menus.append(menu)
                joined.to_csv(OUT/f'{method}_{family}_{load:.2f}_states.csv',index=False)
                menu.to_csv(OUT/f'{method}_{family}_{load:.2f}_menu_bounds.csv',index=False)
                print('PROFILE',json.dumps(stat),flush=True)
            t=time.perf_counter();portfolio,result=aggregate_common(pd.concat(menus));planning_seconds=time.perf_counter()-t
            # Selection is completed before independent reference audit.
            truth=json.loads((WORK/'benchmarks/frozen_v1/normal_common_planning_summary.json').read_text())[family]
            result.update(family=family,method=method,reference_selected_plan=truth['selected_plan'],
                          reference_objective=truth['selected_objective'],
                          selected_plan_matches=result['selected_plan']==truth['selected_plan'])
            result['reference_objective_covered']=result['selected_lower']-1e-8<=truth['selected_objective']<=result['selected_upper']+1e-8
            assert result['certified_optimal'] and result['selected_plan_matches'] and result['reference_objective_covered'],result
            planning.append(result);portfolio.to_csv(OUT/f'{method}_{family}_common_portfolios.csv',index=False)
            online=sum(s['wall_seconds'] for s in ss)
            runtime=dict(family=family,method=method,states=sum(s['states'] for s in ss),
                training_seconds=train,training_samples=sum(r.samples for r in atlas.responses.values()),
                offline_certificate_seconds=offline,online_seconds=online,
                online_certificate_seconds=sum(s['certificate_seconds'] for s in ss),
                menu_aggregation_seconds=menu_seconds,common_selection_seconds=planning_seconds,
                total_seconds=train+offline+online+menu_seconds+planning_seconds,
                certified=sum(s['certified'] for s in ss),classification_errors=sum(s['classification_errors'] for s in ss))
            families.append(runtime);print('FAMILY',json.dumps(runtime),flush=True)
    for family,loads in FAMILIES.items():
        for load in loads:
            records=[p for p in profiles if p['family']==family and p['load']==load]
            assert len(records)==2 and records[0]['prediction_sha256']==records[1]['prediction_sha256'],(family,load,'Prediction mismatch')
    totals=[]
    for method in ['full_current_corrected','compositional']:
        records=[r for r in families if r['method']==method];row=dict(method=method,families=len(records))
        for key in ['states','training_seconds','training_samples','offline_certificate_seconds','online_seconds','online_certificate_seconds','menu_aggregation_seconds','common_selection_seconds','total_seconds','certified','classification_errors']:
            row[key]=sum(r[key] for r in records)
        totals.append(row)
    for row in totals:
        row['speedup_vs_full_current_corrected']=totals[0]['total_seconds']/row['total_seconds']
        row['time_reduction_vs_full_current_corrected']=1-row['total_seconds']/totals[0]['total_seconds']
    assert source_hashes()=={k:v for k,v in hashes.items() if k!='research/normalized_response/certify_study.py'}
    assert hashlib.sha256((HERE/'certify_study.py').read_bytes()).hexdigest()==hashes['research/normalized_response/certify_study.py']
    result=dict(protocol=protocol,totals=totals,family_runtime=families,profiles=profiles,common_planning=planning)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print('TOTAL',json.dumps(totals),flush=True)


if __name__=='__main__':main()
