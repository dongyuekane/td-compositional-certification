"""Same-prediction full-network versus compositional certification ablation.

The 96 states are declared before reference evaluation. Both constructions see
the identical degree-6 response prediction. The full-network baseline uses
residual-centered loss AND electrical-output bounds. No exact AC solve occurs
inside either certification timer. Reference solves follow both certificates.
This measures certificate construction, not an end-to-end planner speedup.
"""
from pathlib import Path
import sys,time,json,itertools,hashlib
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(WORK/'theory'))
sys.path.insert(0,str(WORK/'research'/'control_blocks'))
from atlas import Atlas,Reduced,reference,SEED
from run_study import MainTemplates,FastReduced
from feeder_tube import certify_response
from coupled_certificate import certify_main,slack_loss_interval,classify
from current_certificates import (CurrentModel,current_single_certificate,
    feasibility_certificate)
from corrected_output import (current_residual_enclosure,corrected_state_enclosure,
    corrected_loss_interval)
from certificates import dot_enclosure
import numpy as np
import pandas as pd

OUT=HERE/'full_certificate_ablation_results'

def source_hashes():
    paths=[HERE/'atlas.py',HERE/'coupled_certificate.py',HERE/'run_study.py',
           WORK/'research/control_blocks/feeder_tube.py',WORK/'theory/current_certificates.py',
           WORK/'theory/corrected_output.py']
    return {str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def full_certificate(system,predicted):
    start=time.perf_counter()
    model=CurrentModel(system);x=model.augment(predicted['x'])
    C=np.linalg.inv(model.jacobian(x))
    certificate=current_single_certificate(model,x,C)
    if not certificate['certified']:
        return dict(certified=False,status='unresolved',seconds=time.perf_counter()-start,
                    loss_lower=-np.inf,loss_upper=np.inf,dimension=model.n,
                    reason=certificate.get('reason'))
    f,fr=current_residual_enclosure(model,x)
    r,rr=dot_enclosure(-C,f,fr)
    corrected=corrected_state_enclosure(model,x,C,certificate,r,rr)
    interval=corrected_loss_interval(model,x,C,certificate,correction=r,correction_radius=rr)
    # The correction encloses the already certified root. A new root solve or
    # new existence certificate at the corrected center is unnecessary.
    output_certificate=dict(certified=True,error=corrected['error'])
    status=feasibility_certificate(model,corrected['center'],output_certificate)
    return dict(certified=True,status=status['status'],seconds=time.perf_counter()-start,
                loss_lower=interval['lower'],loss_upper=interval['upper'],dimension=model.n,
                state_error=certificate['error'],corrected_state_error=corrected['error'],
                violation_lower=status['violation_lower'],violation_upper=status['violation_upper'])

def compositional_certificate(reduced,predicted,tubes):
    start=time.perf_counter()
    certificate=certify_main(reduced,predicted,tubes)
    interval=slack_loss_interval(reduced,predicted,certificate)
    status=classify(reduced,predicted,certificate,tubes)
    result=dict(certified=certificate['certified'],status=status['status'],
                seconds=time.perf_counter()-start,loss_lower=interval[0],loss_upper=interval[1],
                dimension=reduced.main.n)
    if certificate['certified']:
        result.update(state_error=certificate['error'],violation_lower=status['violation_lower'],
                      violation_upper=status['violation_upper'])
    else:result['reason']=certificate.get('reason')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    protocol=dict(degree=6,domain=[.4,1.8],shape_seed=SEED,
                  family_loads={'mixed9':1.25,'mixed30':.8},
                  physical_grid={'T':[0.,1.],'X':[0.,1.],'F':[0,1]},
                  control_selection='Every seventh action from the declared normal control menu, starting at zero; six per physical combination.',
                  expected_states=96,
                  same_prediction=True,
                  full_method='Current-injection full-network inclusion, residual correction, quadratic loss at corrected center, electrical enclosures at corrected center.',
                  compositional_method='Offline whole-domain feeder tubes plus main-only inclusion and output composition; static main Hessian cached per T/profile.',
                  timing_scope='Each timer includes fresh preconditioning, regional derivatives and all loss/feasibility outputs. Prediction and offline training/certification are separate. No reference solve or file decompression is inside a certification timer.',
                  state_order='Lexicographic physical/control order. Certificate execution order alternates to reduce order bias.',
                  source_sha256=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    t=time.perf_counter();atlas=Atlas(degree=6,domain=(.4,1.8));training_seconds=time.perf_counter()-t
    tube_by_id={};offline=[];t=time.perf_counter()
    for label,response in atlas.responses.items():
        tube=certify_response(response);assert tube['certified'],(label,tube)
        tube_by_id[id(response)]=tube
        offline.append(dict(label=list(label),training_seconds=response.seconds,
                            training_samples=response.samples,training_newton=response.iterations,
                            certification_seconds=tube['seconds'],power_error=tube['power_error']))
    tube_seconds=time.perf_counter()-t
    records=[];reference_seconds=0.;prediction_seconds=0.;assembly_seconds=0.
    for family,load in [('mixed9',1.25),('mixed30',.8)]:
        templates=MainTemplates()
        for T,X,F in itertools.product((0.,1.),(0.,1.),(0,1)):
            plan=dict(T=T,X=X,F=F,S=0,R=1,C=1)
            for control in list(reference.controls(plan))[::7]:
                t=time.perf_counter()
                system=reference.build(family,load,plan,control,shape_seed=SEED)
                reduced=FastReduced(system,atlas,templates)
                assembly_seconds+=time.perf_counter()-t
                t=time.perf_counter();predicted=reduced.solve();prediction_seconds+=time.perf_counter()-t
                assert predicted['converged'] and not predicted['domain_violation']
                tubes=[tube_by_id[id(r)] for r in reduced.responses]
                if len(records)%2:
                    full=full_certificate(system,predicted)
                    comp=compositional_certificate(reduced,predicted,tubes)
                else:
                    comp=compositional_certificate(reduced,predicted,tubes)
                    full=full_certificate(system,predicted)
                # Validation starts only after both intervals are completed.
                t=time.perf_counter();truth=system.solve(tol=1e-11,maxiter=30)
                metrics=system.metrics(truth['x']);reference_seconds+=time.perf_counter()-t
                assert truth['converged']
                violation=max((.9-metrics['vmin'])/.1,(metrics['vmax']-1.1)/.1,
                              metrics['thermal']-1,metrics['genviol'])
                feasible=violation<=1e-7
                row=dict(family=family,load=load,T=T,X=X,F=F,tap0=control['taps'][0],
                         tap1=control['taps'][1],cap0=control['caps'][0],cap1=control['caps'][1],
                         true_loss=metrics['loss'],true_violation=violation,true_feasible=feasible,
                         prediction_residual=predicted['full_residual'])
                for name,result in [('full',full),('compositional',comp)]:
                    row.update({name+'_'+k:v for k,v in result.items()})
                    if result['certified']:
                        row[name+'_loss_covered']=result['loss_lower']<=metrics['loss']<=result['loss_upper']
                        assert row[name+'_loss_covered'],row
                    if result['status']=='feasible':assert feasible,row
                    if result['status']=='violating':assert not feasible,row
                records.append(row)
    assert len(records)==96
    assert source_hashes()==hashes,'Source changed during ablation; rerun on a stable version.'
    frame=pd.DataFrame(records);frame.to_csv(OUT/'comparisons.csv',index=False)
    summaries=[]
    for family,g in [('all',frame),*frame.groupby('family')]:
        for method in ['full','compositional']:
            summaries.append(dict(family=family,method=method,states=len(g),
                 certified=int(g[method+'_certified'].sum()),
                 feasible=int((g[method+'_status']=='feasible').sum()),
                 violating=int((g[method+'_status']=='violating').sum()),
                 unresolved=int((g[method+'_status']=='unresolved').sum()),
                 total_certificate_seconds=float(g[method+'_seconds'].sum()),
                 median_certificate_seconds=float(g[method+'_seconds'].median()),
                 median_loss_interval_width=float((g[method+'_loss_upper']-g[method+'_loss_lower']).median()),
                 max_dimension=int(g[method+'_dimension'].max())))
    result=dict(protocol=protocol,certificate_comparisons=summaries,
                common_offline=dict(response_training_seconds=training_seconds,
                                    response_training_samples=sum(r.samples for r in atlas.responses.values()),
                                    feeder_certificate_seconds=tube_seconds,labels=len(atlas.responses)),
                shared_prediction_seconds=prediction_seconds,shared_assembly_seconds=assembly_seconds,
                validation_reference_seconds=reference_seconds,offline_curve_records=offline,
                claim_scope='Same predicted states; certificate-construction ablation only. Shared preprocessing and prediction are not free end-to-end speedup.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['offline_curve_records','protocol']},indent=2))

if __name__=='__main__':main()
