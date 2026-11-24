"""Degree-two diagnostic and degree-six refinement on the frozen normal data.

This is a precision stress test, not a speed comparison. All unresolved
actions and investments survive certificate-based selection. Reference data
are used only for the audit, and the already completed degree-six certified
study supplies the uniform refinement result.
"""
from pathlib import Path
import hashlib, json, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'control_blocks'))
import certify_study as cs
from run_study import (Atlas, baseline, ac, np, pd, FAMILIES, SHAPE_SEED,
                       catalog, WORK, atlas_profile, common_result)
from feeder_tube import certify_response

OUT = HERE / 'coarse_precision_results'
OUT.mkdir(exist_ok=True)
cs.OUT = OUT
NONREDUNDANT = {p['plan'] for p in catalog() if p['S'] == 0}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return '+infinity' if value > 0 else '-infinity' if value < 0 else None
    return value


def main():
    started=time.perf_counter();ac.case=baseline.CACHED_CASE
    paths=[HERE/'atlas.py', HERE/'coupled_certificate.py', HERE.parent/'control_blocks/feeder_tube.py', HERE/'run_study.py', HERE/'certify_study.py']
    hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fine_summary=json.loads((HERE/'certified_results/summary.json').read_text())
    report=dict(protocol=dict(purpose='Fixed-data low-precision stress test, not a speed benchmark',
                families=FAMILIES,shape_seed=SHAPE_SEED,coarse_degree=2,refined_degree=6,
                normalized_domain=[.4,1.8],states=3888,
                nonredundant_catalog='S=0 only; 72 legacy-indexed investments. S buys a tie usable only in faults, so S=1 is structurally dominated in this normal-only study.',
                refinement='Uniform degree-six certification from the previously completed full study; files and source hashes are checked, and no reference optimum chooses actions or refinement.',
                semantics='Local high-voltage branch, real-arithmetic certificates with guarded binary64 implementation.',
                source_sha256=hashes),families=[],profiles=[],catalog72_summary=[])
    for family,loads in FAMILIES.items():
        t=time.perf_counter();atlas=Atlas(degree=2,domain=(.4,1.8));train=time.perf_counter()-t
        t=time.perf_counter();tubes={key:certify_response(r) for key,r in atlas.responses.items()};offline=time.perf_counter()-t
        prior=None;plainivs=[];plainstats=[];plainjoins={}
        for load in loads:
            d,iv,stat,prior,joined=atlas_profile(family,load,atlas,2,prior)
            d.to_csv(OUT/f'degree2_{family}_{load:.2f}_uncertified_states.csv',index=False)
            joined.to_csv(OUT/f'degree2_{family}_{load:.2f}_uncertified_audit.csv',index=False)
            plainivs.append(iv);plainstats.append(stat);plainjoins[load]=joined
        plainiv=pd.concat(plainivs)
        plainplan=common_result(plainiv[plainiv.plan.isin(NONREDUNDANT)],family,'degree2_uncertified_72')
        prior=None;stats=[];menus=[];coarseactions=[]
        wrong_approval_withheld=wrong_rejection_retained=0
        for load in loads:
            d,stat,prior,joined=cs.certificate_profile(family,load,atlas,tubes,2,prior)
            menu=cs.menu_bounds(d);menu=menu[menu.plan.isin(NONREDUNDANT)]
            menus.append(menu);stats.append(stat);coarseactions.append(d)
            merged=plainjoins[load][cs.KEYS+['feasible_new','feasible_reference']].merge(d[cs.KEYS+['status']],on=cs.KEYS,validate='one_to_one')
            wrong_approval=merged.feasible_new & ~merged.feasible_reference
            wrong_rejection=~merged.feasible_new & merged.feasible_reference
            approved_withheld=int((wrong_approval & (merged.status!='feasible')).sum())
            rejected_retained=int((wrong_rejection & (merged.status!='violating')).sum())
            wrong_approval_withheld+=approved_withheld;wrong_rejection_retained+=rejected_retained
            stat.update(uncertified_false_approvals=int(wrong_approval.sum()),
                        uncertified_false_rejections=int(wrong_rejection.sum()),
                        wrong_approvals_not_certified=approved_withheld,
                        wrong_rejections_not_deleted=rejected_retained)
            # Exact-reference labels enter this diagnostic CSV only.
            merged.to_csv(OUT/f'degree2_{family}_{load:.2f}_wrong_decision_audit.csv',index=False)
            joined.to_csv(OUT/f'degree2_{family}_{load:.2f}_certified_states.csv',index=False)
            menu.to_csv(OUT/f'degree2_{family}_{load:.2f}_menu_bounds72.csv',index=False)
            print('COARSE PROFILE',json.dumps(finite_json(stat)),flush=True)
        coarseplan=cs.common_bounds(pd.concat(menus),family,2)
        # A smallest-upper-bound incumbent is not an approved investment when
        # rival intervals overlap. Keep those two outcomes unambiguously apart.
        incumbent=coarseplan.get('selected_plan')
        if incumbent is not None:
            exact_portfolios=pd.read_csv(WORK/f'benchmarks/frozen_v1/{family}_normal_common_portfolios.csv').set_index('plan')
            exact_incumbent=float(exact_portfolios.loc[incumbent,'objective'])
            coarseplan['incumbent_reference_objective']=exact_incumbent
            coarseplan['incumbent_reference_objective_covered']=bool(coarseplan['selected_objective_lower']-1e-8<=exact_incumbent<=coarseplan['selected_objective_upper']+1e-8)
            coarseplan.pop('reference_objective_covered',None)
        if not coarseplan['certified_optimal']:
            coarseplan['provisional_incumbent_plan']=incumbent
            coarseplan['selected_plan']=None
            coarseplan['decision']='refinement_required'
            coarseplan.pop('selected_plan_matches',None)
        fine_menus=[];fine_decisions={};fine_count=0
        for load in loads:
            fm=pd.read_csv(HERE/f'certified_results/degree6_{family}_{load:.2f}_menu_bounds.csv')
            fine_menus.append(fm[fm.plan.isin(NONREDUNDANT)])
            fs=pd.read_csv(HERE/f'certified_results/degree6_{family}_{load:.2f}_states.csv')
            assert len(fs)==648 and fs.certified.all()
            fine_count+=len(fs)
            for status,count in fs.status.value_counts().items():fine_decisions[status]=fine_decisions.get(status,0)+int(count)
            cm=pd.concat(coarseactions);cm=cm[cm.load==load]
            comparison=cm[cs.KEYS+['status','certified']].merge(fs[cs.KEYS+['status','certified']],on=cs.KEYS,suffixes=('_degree2','_degree6'),validate='one_to_one')
            comparison.to_csv(OUT/f'{family}_{load:.2f}_uniform_refinement.csv',index=False)
        fineplan=cs.common_bounds(pd.concat(fine_menus),family,6)
        familyreport=dict(family=family,states=1944,training_seconds=train,offline_certificate_seconds=offline,
                          offline_labels=len(tubes),offline_certified=sum(t['certified'] for t in tubes.values()),
                          offline_failures=[dict(label=list(k),reason=t.get('reason')) for k,t in tubes.items() if not t['certified']],
                          uncertified_false_approvals=sum(s['false_feasible'] for s in plainstats),
                          uncertified_false_rejections=sum(s['false_infeasible'] for s in plainstats),
                          uncertified_max_loss_error_mw=max(s['max_loss_error_mw'] for s in plainstats),
                          uncertified_max_complex_voltage_error_pu=max(s['max_complex_voltage_error_pu'] for s in plainstats),
                          uncertified_planning72=plainplan,
                          certified_roots=sum(s['certified'] for s in stats),
                          unresolved_actions=sum(s['feasibility_decisions'].get('unresolved',0) for s in stats),
                          certificate_classification_errors=sum(s['classification_errors'] for s in stats),
                          loss_coverage_failures=sum(s['loss_coverage_failures'] for s in stats),
                          violation_coverage_failures=sum(s['violation_coverage_failures'] for s in stats),
                          wrong_approvals_not_certified=wrong_approval_withheld,
                          wrong_rejections_not_deleted=wrong_rejection_retained,
                          coarse_certified_planning72=coarseplan,
                          uniform_degree6_refinement=dict(states=fine_count,status_counts=fine_decisions,planning72=fineplan))
        report['families'].append(familyreport);report['profiles']+=stats
        reference=pd.read_csv(WORK/f'benchmarks/frozen_v1/{family}_normal_common_portfolios.csv')
        truth72=reference[reference.plan.isin(NONREDUNDANT)]
        truth72.to_csv(OUT/f'{family}_reference_common_portfolios72.csv',index=False)
        reference_best=int(truth72[truth72.feasible].objective.idxmin())
        report['catalog72_summary'].append(dict(family=family,catalog_size=len(truth72),reference_feasible=int(truth72.feasible.sum()),
                    reference_selected_plan=int(truth72.loc[reference_best,'plan']),
                    coarse_uncertified_selected_plan=plainplan.get('selected_plan'),
                    refined_selected_plan=fineplan['selected_plan'],refined_nearest_rival_plan=fineplan['nearest_rival_plan'],
                    refined_certified_optimal=fineplan['certified_optimal'],refined_certified_gap_million_cny=fineplan['certified_gap_million_cny']))
        print('COARSE FAMILY',json.dumps(finite_json(familyreport)),flush=True)
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(WORK))], 'Source changed during experiment.'
    for name,digest in fine_summary['protocol']['source_sha256'].items():
        if name in hashes:assert digest==hashes[name], 'Fine-study source mismatch; do not replay refinement.'
    report['totals']=dict(states=sum(f['states'] for f in report['families']),
            uncertified_false_approvals=sum(f['uncertified_false_approvals'] for f in report['families']),
            uncertified_false_rejections=sum(f['uncertified_false_rejections'] for f in report['families']),
            certified_roots=sum(f['certified_roots'] for f in report['families']),
            unresolved_actions=sum(f['unresolved_actions'] for f in report['families']),
            certificate_classification_errors=sum(f['certificate_classification_errors'] for f in report['families']),
            wrong_approvals_not_certified=sum(f['wrong_approvals_not_certified'] for f in report['families']),
            wrong_rejections_not_deleted=sum(f['wrong_rejections_not_deleted'] for f in report['families']))
    report['audit_wall_seconds']=time.perf_counter()-started
    (OUT/'summary.json').write_text(json.dumps(finite_json(report),indent=2)+'\n')
    print(json.dumps(finite_json(report),indent=2),flush=True)


if __name__=='__main__':main()
