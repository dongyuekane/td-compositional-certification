"""Independent-start adaptive refinement after the shared third-iterate menus.

All decisions use certificate intervals and potential-minimizer sets. Full
reference values are used only to assess the resulting signs. Each method and
each investment pair starts from a fresh copy of the same prepared records;
no newly refined state propagates between pairs. Refinements really solve AC
from the flat start and recompute the full-state certificate and feasibility.
"""
from __future__ import annotations
from pathlib import Path
import sys,copy,json,time,hashlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'simulation'))
from menu_study import (StateBank,compare,CurrentModel,current_single_certificate,
    feasibility_certificate,quantity_interval,catalog,LOSS_FACTOR,build)
from scipy.linalg import solve
import numpy as np
import pandas as pd

SOURCE=HERE.parent/'results/menu_study'
OUT=HERE/'adaptive_refinement_results'
OUT.mkdir(exist_ok=True)
DATA=HERE/'pilot'


class FreshBank(StateBank):
    def __init__(self,template):
        # Tables/traces are immutable input data. Only certificate state and
        # caches are copied; no prior method's new physical solution is shared.
        self.stem=template.stem;self.table=template.table;self.trace=template.trace
        self.records=copy.deepcopy(template.records);self.active=dict(template.active)
        self.cache={};self.seconds=0.;self.refined=set(template.refined)
        self.initial_refined=set(self.refined);self.fresh_x={};self.refinement_rows=[]

    def reconstruct(self,idx,k=None):
        s,m,x=super().reconstruct(idx,k)
        step=self.active.get(idx,2) if k is None else k
        if step==3 and idx in self.fresh_x:x=m.augment(self.fresh_x[idx])
        return s,m,x

    def compute(self,idx,full=True):
        assert full,"This audit only adds complete refinements to prepared menus."
        started=time.perf_counter();cpu0=time.process_time()
        r=self.table.iloc[idx]
        plan=dict(T=float(r['T']),X=float(r.X),F=int(r.F),S=1,R=1,C=1)
        control=dict(taps=(r.tap0,r.tap1),caps=(r.cap0,r.cap1),service=r.service,tie=int(r.tie))
        t0=time.perf_counter()
        s=build(r.family,r.load,plan,control,event=int(r.event),shape_seed=int(r.shape_seed))
        physical=s.solve(x0=None,tol=1e-10,maxiter=30,trace=True)
        physical_seconds=time.perf_counter()-t0
        assert physical['converged']
        m=CurrentModel(s);x=m.augment(physical['x'])
        t0=time.perf_counter();C=np.linalg.inv(m.jacobian(x));certificate=current_single_certificate(m,x,C)
        fc=feasibility_certificate(m,x,certificate)
        qi=quantity_interval(m.loss_quantity(),x,certificate['error']) if certificate['certified'] else dict(lower=-np.inf,upper=np.inf,error=np.inf)
        certificate_seconds=time.perf_counter()-t0
        row=dict(idx=int(idx),checkpoint='converged',state_certified=certificate['certified'],status=fc['status'],lower=qi['lower'],upper=qi['upper'],error=qi['error'],seconds=time.perf_counter()-started)
        self.fresh_x[idx]=physical['x'].copy();self.active[idx]=3;self.refined.add(idx);self.records[idx]=row
        self.cache[(idx,3)]=dict(s=s,m=m,x=x,C=C,q=m.loss_quantity())
        self.seconds+=row['seconds']
        self.refinement_rows.append(dict(stem=str(self.stem.relative_to(HERE)),idx=int(idx),physical_seconds=physical_seconds,certificate_seconds=certificate_seconds,total_seconds=row['seconds'],cpu_seconds=time.process_time()-cpu0,newton_iterations=physical['iterations'],state_certified=certificate['certified'],status=fc['status']))
        # Evaluation-only consistency assertions; neither controls nor stopping
        # decisions use these full-reference fields.
        assert abs(physical['loss']-r.loss)<1e-8
        if certificate['certified']:assert qi['lower']<=r.loss<=qi['upper']
        if fc['status']=='feasible':assert bool(r.feasible)
        if fc['status']=='violating':assert not bool(r.feasible)
        return row


def template(stem,record_file):
    bank=StateBank(stem)
    # Materialize immutable checkpoints once, outside all online timings. An
    # NpzFile __getitem__ otherwise decompresses the entire array on every row.
    # This is shared input loading, not reuse of newly refined physical states.
    checkpoint_array=bank.trace['x'];bank.trace.close();bank.trace={'x':checkpoint_array}
    rows=pd.read_csv(record_file)
    for r in rows.to_dict('records'):
        idx=int(r['idx']);r['idx']=idx;bank.records[idx]=r
        bank.active[idx]=3 if str(r['checkpoint'])=='converged' else 2
        if bank.active[idx]==3:bank.refined.add(idx)
    assert len(bank.records)==len(bank.table)
    return bank


def sign(lo,hi):return 1 if lo>0 else -1 if hi<0 else 0


def bounds(method,bi,mi,bj,mj,kind):
    if mi['status']!='certified' or mj['status']!='certified':
        return dict(lower=-np.inf,upper=np.inf,sign=0,pair_calls=0,status='unresolved')
    if kind=='fault' and abs(mi['service']-mj['service'])>1e-8:
        return dict(lower=-np.inf,upper=np.inf,sign=0,pair_calls=0,status='different_service_tiers')
    pair_calls=0;fallback=0
    il,iu=mi['lower']-mj['upper'],mi['upper']-mj['lower']
    if kind=='normal':
        delta_capital=mi['capital']-mj['capital']
        initial_sign=sign(delta_capital+LOSS_FACTOR*il,delta_capital+LOSS_FACTOR*iu)
    else:initial_sign=sign(il,iu)
    # Deterministic guard: a paired certificate cannot add decision value once
    # the already-available independent interval has established the sign.
    if method=='paired' and initial_sign:
        lo,hi=il,iu
        if kind=='normal':lo,hi=delta_capital+LOSS_FACTOR*lo,delta_capital+LOSS_FACTOR*hi
        return dict(lower=float(lo),upper=float(hi),sign=initial_sign,pair_calls=0,fallbacks=0,status='certified',independent_guard=True)
    if method=='paired':
        pair=compare(bi,mi,bj,mj);lo,hi=pair['pair_lower'],pair['pair_upper']
        pair_calls=pair['pair_calls'];fallback=pair['independent_fallbacks']
    else:lo,hi=il,iu
    if kind=='normal':
        capital=mi['capital']-mj['capital'];lo,hi=capital+LOSS_FACTOR*lo,capital+LOSS_FACTOR*hi
    return dict(lower=float(lo),upper=float(hi),sign=sign(lo,hi),pair_calls=pair_calls,fallbacks=fallback,status='certified')


def one_pair(method,kind,ti,mi0,tj,mj0,identity):
    started=time.perf_counter();cpu0=time.process_time()
    bi=FreshBank(ti);bj=bi if ti is tj else FreshBank(tj)
    mi,mj=copy.deepcopy(mi0),copy.deepcopy(mj0)
    pmap={p['plan']:p for p in catalog()}
    initial_menu_states=len(set(mi['ids'])|set(mj['ids'])) if bi is bj else len(mi['ids'])+len(mj['ids'])
    initial_full_states=sum(bi.active[i]==3 for i in mi['ids'])+sum(bj.active[i]==3 for i in mj['ids'])
    comparisons=[];refinement_rounds=0;bound_seconds=0.
    while True:
        t0=time.perf_counter();b=bounds(method,bi,mi,bj,mj,kind);bound_seconds+=time.perf_counter()-t0
        comparisons.append(dict(round=refinement_rounds,P_i=len(mi.get('P',[])),P_j=len(mj.get('P',[])),**b))
        if b['sign'] or b['status']=='different_service_tiers':break
        if mi['status']!='certified' or mj['status']!='certified':break
        todo=[]
        for bank,menu in [(bi,mi),(bj,mj)]:
            for idx in menu['P']:
                if bank.active[idx]!=3 and not any(bank is bb and idx==jj for bb,jj in todo):todo.append((bank,idx))
        if not todo:break
        for bank,idx in todo:bank.compute(idx,True)
        refinement_rounds+=1
        # Exactly the same declared control menus are rebuilt; full-reference
        # actions do not select the candidates that receive refinement.
        mi=bi.menu(pmap[mi0['plan']],ids=mi0['ids'],refine=False)
        mj=bj.menu(pmap[mj0['plan']],ids=mj0['ids'],refine=False)
    refinement=bi.refinement_rows+(bj.refinement_rows if bi is not bj else [])
    truth=(mi0['truth_loss']-mj0['truth_loss'])
    if kind=='normal':truth=mi0['capital']-mj0['capital']+LOSS_FACTOR*truth
    truth_sign=sign(truth,truth)
    covered=bool(b['lower']<=truth<=b['upper'])
    correct=bool(b['sign']==truth_sign) if b['sign'] else None
    assert covered and (correct is None or correct)
    row=dict(kind=kind,method=method,**identity,plan_i=mi0['plan'],plan_j=mj0['plan'],
             initial_menu_states=initial_menu_states,initial_menu_full_states=initial_full_states,
             initial_sign_decided=bool(comparisons[0]['sign']),refinement_rounds=refinement_rounds,
             additional_full_states=len(refinement),additional_cold_newton_iterations=sum(r['newton_iterations'] for r in refinement),
             additional_physical_seconds=sum(r['physical_seconds'] for r in refinement),
             additional_single_certificate_seconds=sum(r['certificate_seconds'] for r in refinement),
             refinement_seconds=sum(r['total_seconds'] for r in refinement),bound_seconds=bound_seconds,
             pair_certificate_calls=sum(r['pair_calls'] for r in comparisons),
             final_sign_decided=bool(b['sign']),final_sign=b['sign'],true_sign=truth_sign,
             true_difference=float(truth),final_lower=b['lower'],final_upper=b['upper'],
             final_coverage=covered,final_correct=correct,online_additional_seconds=time.perf_counter()-started,
             online_additional_cpu_seconds=time.process_time()-cpu0)
    return row,dict(**identity,kind=kind,method=method,plan_i=mi0['plan'],plan_j=mj0['plan'],comparisons=comparisons,refinements=refinement)


def initialization_physics(templates):
    """Measure the same three cold Newton updates for every prepared state.

    This is initialization accounting, not further reference accuracy or a
    scenario expansion. Models are rebuilt as in the actual replay pipeline.
    """
    start=time.perf_counter();cpu0=time.process_time();updates=0;full=0
    for bank in templates:
        for i,a in enumerate(bank.table.itertuples()):
            plan=dict(T=a.T,X=a.X,F=int(a.F),S=1,R=1,C=1)
            c=dict(taps=(a.tap0,a.tap1),caps=(a.cap0,a.cap1),service=a.service,tie=int(a.tie))
            s=build(a.family,a.load,plan,c,int(a.event),int(a.shape_seed))
            x=s.flat()
            for _ in range(3):x+=solve(s.jacobian(x),-s.residual(x),check_finite=False);updates+=1
            assert np.max(abs(x-bank.trace['x'][i,2,:s.n]))<1e-10
            if bank.active[i]==3:
                # Preserve the shared initialization's fallback requirement.
                rr=s.solve(x0=None,tol=1e-10,maxiter=30);assert rr['converged'];full+=1
    return dict(states=sum(len(t.table) for t in templates),three_step_newton_updates=updates,initial_full_fallback_states=full,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0)


def main():
    started=time.perf_counter()
    sourcefiles=[SOURCE/'normal_menus.json',SOURCE/'normal_pairs.csv',SOURCE/'normal_summary.json',SOURCE/'fault_summary.json',SOURCE/'fault_pairs.csv']+list(SOURCE.glob('*_state_certificates.csv'))
    source_hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles}
    codefiles=[HERE.parent/'theory/current_certificates.py',HERE.parent/'theory/certificates.py',HERE.parent/'simulation/menu_study.py']
    source_hashes.update({str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in codefiles})
    normalmenus=json.loads((SOURCE/'normal_menus.json').read_text());faultsummary=json.loads((SOURCE/'fault_summary.json').read_text())
    normalpairs=pd.read_csv(SOURCE/'normal_pairs.csv');faultpairs=pd.read_csv(SOURCE/'fault_pairs.csv')
    assert len(normalpairs)==198 and len(faultpairs)==8
    menus={(m['family'],m['load'],m['plan']):m for m in normalmenus}
    templates={};allrows=[];details=[]
    for fam,load in sorted(set((m['family'],m['load']) for m in normalmenus)):
        templates[(fam,load)]=template(DATA/'unmodified'/f'{fam}_{load:.2f}',SOURCE/f'{fam}_{load:.2f}_state_certificates.csv')
    faultmenus={}
    for r in faultsummary:
        key=(r['family'],r['load'],r['event'],r['plan']);fam,load,event,p=key
        templates[key]=template(DATA/'faults'/f'{fam}_{load:.2f}_p{p}_e{event}',SOURCE/f'{fam}_{load:.2f}_p{p}_e{event}_state_certificates.csv')
        faultmenus[key]=r['menu']
    init=initialization_physics(list(templates.values()));print('INITIALIZATION PHYSICS',json.dumps(init),flush=True)
    for r in normalpairs.itertuples():
        t=templates[(r.family,r.load)];mi=menus[(r.family,r.load,int(r.plan_i))];mj=menus[(r.family,r.load,int(r.plan_j))]
        for method in ['independent','paired']:
            row,detail=one_pair(method,'normal',t,mi,t,mj,dict(family=r.family,load=r.load,event=0));allrows.append(row);details.append(detail)
        if len(allrows)%40==0:print('NORMAL PROGRESS',len(allrows)//2,flush=True)
    for r in faultpairs.itertuples():
        ki=(r.family,r.load,int(r.event),47);kj=(r.family,r.load,int(r.event),31)
        for method in ['independent','paired']:
            row,detail=one_pair(method,'fault',templates[ki],faultmenus[ki],templates[kj],faultmenus[kj],dict(family=r.family,load=r.load,event=int(r.event)));allrows.append(row);details.append(detail)
    frame=pd.DataFrame(allrows);frame.to_csv(OUT/'comparisons.csv',index=False)
    (OUT/'comparison_details.json').write_text(json.dumps(details,indent=2)+"\n")
    summary=[]
    for (kind,method),g in frame.groupby(['kind','method']):
        summary.append(dict(kind=kind,method=method,pairs=len(g),initial_decided=int(g.initial_sign_decided.sum()),additional_full_states=int(g.additional_full_states.sum()),additional_cold_newton_iterations=int(g.additional_cold_newton_iterations.sum()),final_decided=int(g.final_sign_decided.sum()),unresolved=int((~g.final_sign_decided).sum()),incorrect_decisions=int((g.final_correct==False).sum()),final_coverage=int(g.final_coverage.sum()),additional_physical_seconds=float(g.additional_physical_seconds.sum()),additional_single_certificate_seconds=float(g.additional_single_certificate_seconds.sum()),bound_seconds=float(g.bound_seconds.sum()),online_additional_seconds=float(g.online_additional_seconds.sum()),online_additional_cpu_seconds=float(g.online_additional_cpu_seconds.sum()),pair_certificate_calls=int(g.pair_certificate_calls.sum())))
    normalinit=json.loads((SOURCE/'normal_summary.json').read_text())
    prior_single_seconds=sum(r['single_stage_seconds'] for r in normalinit)+sum(r['single_stage_seconds'] for r in faultsummary)
    common_init_seconds=init['seconds']+prior_single_seconds
    totals=[]
    for method,g in frame.groupby('method'):
        totals.append(dict(method=method,common_initialization_physics_seconds=init['seconds'],common_initialization_saved_single_stage_seconds=prior_single_seconds,additional_online_seconds=float(g.online_additional_seconds.sum()),composed_total_seconds=common_init_seconds+float(g.online_additional_seconds.sum()),additional_full_states=int(g.additional_full_states.sum())))
    changed=[str(p) for p in sourcefiles if hashlib.sha256(p.read_bytes()).hexdigest()!=source_hashes[str(p.relative_to(HERE.parent))]]
    assert not changed,('Source changed during audit; rerun from a stable generation.',changed)
    result=dict(protocol="Independent starts for every pair and method; shared checkpoint-3 state/feasibility/loss certificates; paired method first returns an already decisive independent interval, otherwise computes paired bounds; if still unresolved refine every remaining potential minimizer on both sides, cold flat AC plus full certificate; no oracle-based action choice or cross-pair propagation",normal_decision="Sign of capital increment plus discounted normal-loss difference",fault_decision="Sign of optimal loss difference only, at independently certified equal served MW; no annualized fault-investment economics",initialization=init,previous_single_stage_seconds=prior_single_seconds,method_summary=summary,composed_costs=totals,timing_scope="Initial physics was freshly timed; previous single-stage durations are the saved replay's observed wall times (including its own checkpoint reconstruction costs). Composed totals include these costs and all additional refinement/certificate/bound work, but are not a single uninterrupted runtime comparison. Input checkpoints are materialized before online timings; newly computed states never propagate between pairs or methods. Additional CPU is measured directly. Saved reference solutions are used only to assess accuracy.",source_hashes=source_hashes,total_audit_wall_seconds=time.perf_counter()-started)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
