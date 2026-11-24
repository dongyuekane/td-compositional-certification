"""Full normal-menu audit of composed feeder/main certificates.

Reference solutions enter only the post-timing audit. Every candidate action
is solved and certified, and all portfolio lower/upper bounds are formed from
certificates alone. A local high-voltage-branch certificate is used; numerical
guards are binary64 estimates, not an outward-rounded interval implementation.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'control_blocks'))
from run_study import (Atlas, baseline, ac, np, pd, NetworkReuse, MainTemplates,
                       FastReduced, snake, state_key, FAMILIES, SHAPE_SEED,
                       TAPS, CAPS, catalog, LOSS_FACTOR, WORK, atlas_profile,
                       common_result)
from feeder_tube import certify_response
from coupled_certificate import certify_main, classify, slack_loss_interval

OUT = HERE / 'certified_results'
OUT.mkdir(exist_ok=True)
KEYS = ['T', 'X', 'F', 'tap0', 'tap1', 'cap0', 'cap1']


def certificate_profile(family, load, atlas, tubes, degree, previous_profile=None):
    started = time.perf_counter(); cpu0 = time.process_time()
    maker = NetworkReuse(family, load); templates = MainTemplates()
    last = None; solutions = {}; rows = []
    physical_grid = snake(((0., .5, 1.), (0., .5, 1.), (0, 1)))
    controls = snake((TAPS, TAPS, CAPS, CAPS))
    for gi, (T, X, F) in enumerate(physical_grid):
        p = dict(T=T, X=X, F=F, S=1, R=1, C=1)
        for ta, tb, ca, cb in (controls if gi % 2 == 0 else controls[::-1]):
            c = dict(taps=(ta, tb), caps=(ca, cb), service=1., tie=-1)
            key = state_key(p, c)
            t = time.perf_counter()
            s = maker.system(p, c); model = FastReduced(s, atlas, templates)
            assembly = time.perf_counter() - t
            t = time.perf_counter()
            starts = [('flat', model.main.flat())]
            if last is not None:
                starts.append(('adjacent_control_or_investment', last))
            if previous_profile is not None and key in previous_profile:
                starts.append(('previous_profile_same_state', previous_profile[key]))
            source, x0 = min(starts, key=lambda a: float(max(abs(model.residual(a[1])))))
            warm = time.perf_counter() - t
            t = time.perf_counter(); r = model.solve(x0)
            solve_seconds = time.perf_counter() - t
            # A certificate failure remains unresolved. No truth-based fallback
            # or action deletion is permitted in this certification audit.
            cs = [tubes[k, X, F if k == 0 else 0, c['taps'][k], c['caps'][k]] for k in (0, 1)]
            t = time.perf_counter(); cert = certify_main(model, r, cs)
            main_cert_seconds = time.perf_counter() - t
            t = time.perf_counter(); result = classify(model, r, cert, cs)
            classify_seconds = time.perf_counter() - t
            t = time.perf_counter(); lower, upper = slack_loss_interval(model, r, cert)
            loss_cert_seconds = time.perf_counter() - t
            last = r['main_x'].copy(); solutions[key] = last
            rows.append(dict(family=family, load=load, degree=degree, T=T, X=X, F=F,
                             tap0=ta, tap1=tb, cap0=ca, cap1=cb,
                             converged=r['converged'], domain_violation=r['domain_violation'],
                             certified=cert['certified'], status=result['status'],
                             certificate_reason=cert.get('reason', ''),
                             loss_lower=lower, loss_upper=upper, loss_width=upper-lower,
                             violation_lower=result.get('violation_lower', -np.inf),
                             violation_upper=result.get('violation_upper', np.inf),
                             main_error=cert.get('error', np.inf),
                             max_voltage_error=result.get('max_voltage_error', np.inf),
                             main_dimension=model.main.n, iterations=r['iterations'],
                             initial_source=source, full_residual=r['full_residual'],
                             reduced_residual=r['reduced_residual'], zmin=r['zmin'], zmax=r['zmax'],
                             assembly_seconds=assembly, warm_selection_seconds=warm,
                             solve_seconds=solve_seconds, main_certificate_seconds=main_cert_seconds,
                             classify_seconds=classify_seconds, loss_certificate_seconds=loss_cert_seconds))
    wall = time.perf_counter() - started; cpu = time.process_time() - cpu0
    d = pd.DataFrame(rows)
    old = pd.read_csv(WORK / f'benchmarks/pilot/unmodified/{family}_{load:.2f}_states.csv')
    joined = d.merge(old, on=KEYS, suffixes=('', '_reference'), validate='one_to_one', sort=False)
    assert len(joined) == 648
    joined['loss_covered'] = (joined.loss >= joined.loss_lower-1e-9) & (joined.loss <= joined.loss_upper+1e-9)
    joined['violation_covered'] = (joined.violation >= joined.violation_lower-1e-8) & (joined.violation <= joined.violation_upper+1e-8)
    joined['correct_decision'] = ((joined.status == 'feasible') & joined.feasible) | ((joined.status == 'violating') & ~joined.feasible) | (joined.status == 'unresolved')
    stat = dict(family=family, load=load, degree=degree, states=len(d), wall_seconds=wall,
                cpu_seconds=cpu, certified=int(d.certified.sum()),
                feasibility_decisions=d.status.value_counts().to_dict(),
                classification_errors=int((~joined.correct_decision).sum()),
                loss_coverage_failures=int((~joined.loss_covered).sum()),
                violation_coverage_failures=int((~joined.violation_covered).sum()),
                max_loss_interval_width_mw=float(d.loss_width.max()),
                max_voltage_bound_pu=float(d.max_voltage_error.max()),
                max_main_error=float(d.main_error.max()),
                max_full_residual=float(d.full_residual.max()),
                iterations=int(d.iterations.sum()), domain_violations=int(d.domain_violation.sum()),
                main_templates=len(templates.templates),
                normalized_load_min=float(d.zmin.min()), normalized_load_max=float(d.zmax.max()))
    for name in ['assembly_seconds', 'warm_selection_seconds', 'solve_seconds',
                 'main_certificate_seconds', 'classify_seconds', 'loss_certificate_seconds']:
        stat[name] = float(d[name].sum())
    stat['certificate_only_seconds'] = sum(stat[x] for x in ['main_certificate_seconds', 'classify_seconds', 'loss_certificate_seconds'])
    assert stat['classification_errors'] == stat['loss_coverage_failures'] == stat['violation_coverage_failures'] == 0, stat
    return d, stat, solutions, joined


def menu_bounds(d):
    rows = []
    for p in catalog():
        g = d[(d['T'] == p['T']) & (d.X == p['X']) & (d.F == p['F'])]
        if not p['R']:
            g = g[(g.tap0 == 1.04) & (g.tap1 == 1.04)]
        if not p['C']:
            g = g[(g.cap0 == 0.) & (g.cap1 == 0.)]
        possible = g[g.status != 'violating']; feasible = g[g.status == 'feasible']
        status = 'feasible' if len(feasible) else 'violating' if not len(possible) else 'unresolved'
        lower = float(possible.loss_lower.min()) if len(possible) else np.inf
        upper = float(feasible.loss_upper.min()) if len(feasible) else np.inf
        rows.append(dict(family=d.family.iloc[0], load=d.load.iloc[0], degree=int(d.degree.iloc[0]),
                         plan=p['plan'], capital=p['capital'], status=status,
                         possible_actions=len(possible), certified_feasible_actions=len(feasible),
                         loss_lower=lower, loss_upper=upper))
    return pd.DataFrame(rows)


def common_bounds(menus, family, degree):
    records = []
    for pid, g in menus.groupby('plan'):
        assert len(g) == 3
        status = 'violating' if (g.status == 'violating').any() else 'feasible' if (g.status == 'feasible').all() else 'unresolved'
        lo = float(g.capital.iloc[0] + LOSS_FACTOR*g.loss_lower.mean())
        hi = float(g.capital.iloc[0] + LOSS_FACTOR*g.loss_upper.mean())
        records.append(dict(plan=int(pid), status=status, lower=lo, upper=hi))
    portfolios = pd.DataFrame(records)
    candidates = portfolios[portfolios.status == 'feasible']
    if len(candidates):
        winner = candidates.loc[candidates.upper.idxmin()]
        rivals = portfolios[(portfolios.plan != winner.plan) & (portfolios.status != 'violating')]
        runner = rivals.loc[rivals.lower.idxmin()]
        gap = float(runner.lower-winner.upper)
        selected = int(winner.plan)
        outcome = dict(selected_plan=selected, certified_optimal=bool(gap > 0),
                       selected_objective_lower=float(winner.lower), selected_objective_upper=float(winner.upper),
                       nearest_rival_plan=int(runner.plan), nearest_rival_lower=float(runner.lower),
                       certified_gap_million_cny=gap)
        reference = json.loads((WORK/'benchmarks/frozen_v1/normal_common_planning_summary.json').read_text())[family]
        outcome.update(reference_selected_plan=reference['selected_plan'], selected_plan_matches=selected == reference['selected_plan'],
                       reference_objective=reference['selected_objective'],
                       reference_objective_covered=bool(winner.lower-1e-8 <= reference['selected_objective'] <= winner.upper+1e-8))
    else:
        outcome = dict(selected_plan=None, certified_optimal=False)
    outcome.update(family=family, degree=degree, common_status_counts=portfolios.status.value_counts().to_dict())
    portfolios.to_csv(OUT/f'degree{degree}_{family}_common_portfolio_bounds.csv', index=False)
    return outcome


def main(degrees):
    started = time.perf_counter(); ac.case = baseline.CACHED_CASE
    paths = [HERE/'atlas.py', HERE/'coupled_certificate.py', HERE.parent/'control_blocks/feeder_tube.py', HERE/'run_study.py']
    hashes = {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = dict(families=FAMILIES, shape_seed=SHAPE_SEED, degrees=degrees, labels_per_family=54,
                    normalized_domain=[.4, 1.8], normal_states_per_profile=648,
                    reference_role='Post-timing coverage/classification and selected-investment audit only; no reference action or value enters construction or selection.',
                    certificate_scope='Local coupled high-voltage branch; real-arithmetic certificates implemented with guarded binary64 estimates, not outward-rounded interval arithmetic.',
                    training_cost='Every family independently constructs and certifies all 54 responses. The same Response objects feed its no-certificate and composed-certificate comparison.',
                    timing_scope='Training, offline certificates, and online enumeration including assembly, residual-selected warm starts, reduced solves, constraint/loss certificates. Reference audit and CSV I/O excluded. Menu aggregation separately measured.',
                    main_cache='One main ACSystem per T and profile, retaining cached Hessian row sums.', source_sha256=hashes)
    runtimes=[]; profiles=[]; planning=[]
    for family, loads in FAMILIES.items():
        prior=None; ss=[]; ivs=[]
        for load in loads:
            _,iv,stat,prior=baseline.profile(family,load,'sparse_reuse_warm',prior)
            ss.append(stat);ivs.append(iv)
        runtimes.append(dict(family=family,method='sparse_reuse_warm',degree=None,training_seconds=0.,offline_certificate_seconds=0.,online_seconds=sum(s['wall_seconds'] for s in ss),certificate_only_seconds=0.,total_seconds=sum(s['wall_seconds'] for s in ss),states=sum(s['states'] for s in ss)))
        print('BASELINE',json.dumps(runtimes[-1]),flush=True)
    for degree in degrees:
        for family, loads in FAMILIES.items():
            t=time.perf_counter(); atlas=Atlas(degree=degree,domain=(.4,1.8)); train=time.perf_counter()-t
            t=time.perf_counter(); tubes={key:certify_response(r) for key,r in atlas.responses.items()}; offline=time.perf_counter()-t
            assert all(c['certified'] for c in tubes.values()), 'A feeder label is uncertified.'
            # Identical polynomial responses/cache policy, without certificates.
            prior=None; plainstats=[]
            for load in loads:
                _,_,stat,prior,_=atlas_profile(family,load,atlas,degree,prior);plainstats.append(stat)
            runtimes.append(dict(family=family,method='response_uncertified',degree=degree,training_seconds=train,offline_certificate_seconds=0.,online_seconds=sum(s['wall_seconds'] for s in plainstats),certificate_only_seconds=0.,total_seconds=train+sum(s['wall_seconds'] for s in plainstats),states=sum(s['states'] for s in plainstats)))
            prior=None; stats=[]; menus=[]
            for load in loads:
                d,stat,prior,joined=certificate_profile(family,load,atlas,tubes,degree,prior)
                stats.append(stat);profiles.append(stat)
                t=time.perf_counter();menu=menu_bounds(d);aggregation=time.perf_counter()-t
                stat['menu_aggregation_seconds']=aggregation;menus.append(menu)
                joined.to_csv(OUT/f'degree{degree}_{family}_{load:.2f}_states.csv',index=False)
                menu.to_csv(OUT/f'degree{degree}_{family}_{load:.2f}_menu_bounds.csv',index=False)
                print('CERTIFIED PROFILE',json.dumps(stat),flush=True)
            t=time.perf_counter(); plan=common_bounds(pd.concat(menus),family,degree);planning_seconds=time.perf_counter()-t
            planning.append(plan);print('COMMON PLANNING',json.dumps(plan),flush=True)
            online=sum(s['wall_seconds'] for s in stats)
            runtimes.append(dict(family=family,method='response_composed_certificate',degree=degree,training_seconds=train,offline_certificate_seconds=offline,online_seconds=online,certificate_only_seconds=sum(s['certificate_only_seconds'] for s in stats),total_seconds=train+offline+online,states=sum(s['states'] for s in stats),offline_labels=len(tubes),offline_certified=sum(t['certified'] for t in tubes.values()),menu_aggregation_seconds=sum(s['menu_aggregation_seconds'] for s in stats),common_planning_seconds=planning_seconds,training_samples=sum(r.samples for r in atlas.responses.values()),training_newton_updates=sum(r.iterations for r in atlas.responses.values())))
            print('CERTIFIED FAMILY',json.dumps(runtimes[-1]),flush=True)
    totals=[]
    for method,degree in [('sparse_reuse_warm',None)]+[(m,d) for d in degrees for m in ['response_uncertified','response_composed_certificate']]:
        rr=[r for r in runtimes if r['method']==method and r['degree']==degree]
        row=dict(method=method,degree=degree,families=len(rr))
        for name in ['states','training_seconds','offline_certificate_seconds','online_seconds','certificate_only_seconds','total_seconds']:
            row[name]=sum(r[name] for r in rr)
        totals.append(row)
    for r in totals:
        r['speedup_vs_sparse_uncertified']=totals[0]['total_seconds']/r['total_seconds']
        r['time_reduction_vs_sparse_uncertified']=1-r['total_seconds']/totals[0]['total_seconds']
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(WORK))], 'Source changed during experiment; rerun.'
    summary=dict(protocol=protocol,totals=totals,family_runtime=runtimes,profile_statistics=profiles,common_planning=planning,audit_wall_seconds=time.perf_counter()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--degrees',nargs='+',type=int,default=[6]);args=parser.parse_args()
    main(args.degrees)
