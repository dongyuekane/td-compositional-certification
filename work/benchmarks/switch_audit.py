"""Audit every deduplicated normal investment pair with changed optimal controls.

The tested shortcut freezes the smaller transformer's exact optimal action at
the larger transformer. This is a best-case physical full-solution comparison;
there is no approximation or surrogate error to confound control-switch error.
"""
from reference import *
from independent_audit import independent_metrics
from pypower.api import runpf, ppoption


def action(a):
    return dict(taps=(float(a.tap0),float(a.tap1)),caps=(float(a.cap0),float(a.cap1)),service=1.,tie=-1)


def corridor_outage_audit():
    result=[]
    for family,load,ids,outage in [("mixed9",1.1,[2,4],2),("mixed30",.7,[5,9],9)]:
        plan0=dict(T=0.,X=.5,F=1,S=1,R=1,C=1)
        plan1=dict(plan0,T=1.)
        control=dict(taps=(1.075,1.075),caps=(1.2,1.2),service=1.,tie=-1)
        baseline=build(family,load,plan0,control,1,20261108)
        upgraded=build(family,load,plan1,control,1,20261108)
        inactive_zero_flow_matrix=bool(np.max(abs(upgraded.Yf[outage]))==0 and np.max(abs(upgraded.Yt[outage]))==0)
        active_upgrade=[i for i in ids if i!=outage]
        assert baseline.ppc["branch"][outage,10]==0 and upgraded.ppc["branch"][outage,10]==0 and inactive_zero_flow_matrix
        for i in ids:
            assert np.allclose(upgraded.ppc["branch"][i,2:4],.7*baseline.ppc["branch"][i,2:4])
            assert np.allclose(upgraded.ppc["branch"][i,5:8],1.5*baseline.ppc["branch"][i,5:8])
        assert all(upgraded.ppc["branch"][i,10]==1 for i in active_upgrade)
        r=upgraded.solve();assert r["converged"]
        v=r["voltage"];flow=abs(v[int(upgraded.ppc["branch"][outage,0])-1]*np.conj(upgraded.Yf[outage]@v)*upgraded.base)
        assert flow==0.
        capital0=sum(COST[k]*plan0[k] for k in COST);capital1=sum(COST[k]*plan1[k] for k in COST)
        assert abs((capital1-capital0)-COST["T"])<1e-12
        result.append(dict(family=family,corridor_zero_based_indices=ids,outaged_zero_based_index=outage,active_upgraded_indices=active_upgrade,outaged_branch_flow_mva=float(flow),capital_increment_million_cny=capital1-capital0,capital_retained_when_asset_outaged=True,passed=True))
    return result


def main():
    source=HERE/"pilot/unmodified";out=HERE/"frozen_v1"
    pairs=pd.read_csv(source/"transformer_increment_pairs.csv")
    chosen=pairs[(pairs.S==0)&pairs.feasible_a&pairs.feasible_b&((pairs.tap_switch==True)|(pairs.cap_switch==True))]
    assert len(chosen)==11
    rows=[];catalogue={c["plan"]:c for c in catalog()}
    for p in chosen.itertuples():
        d=pd.read_csv(source/f"{p.family}_{p.load:.2f}_states.csv")
        a=d.loc[int(p.state_a)];b=d.loc[int(p.state_b)]
        fixed=d[(d["T"]==p.T)&(d.X==p.X_b)&(d.F==p.F)&(d.tap0==a.tap0)&(d.tap1==a.tap1)&(d.cap0==a.cap0)&(d.cap1==a.cap1)]
        assert len(fixed)==1;f=fixed.iloc[0]
        exact_delta=p.capital_increment+LOSS_FACTOR*(b.loss-a.loss)
        frozen_delta=p.capital_increment+LOSS_FACTOR*(f.loss-a.loss)
        assert abs(exact_delta-p.objective_difference)<1e-10
        system=build(p.family,p.load,catalogue[int(p.plan_b)],action(a),0,int(p.shape_seed))
        reference,ok=runpf(system.ppc,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-11));assert ok
        check=independent_metrics(reference)
        assert abs(check["loss"]-f.loss)<1e-7 and check["safe"]==bool(f.feasible)
        components=dict(undervoltage=(.9-f.vmin)/.1,overvoltage=(f.vmax-1.1)/.1,thermal=f.thermal-1,generator=f.genviol)
        row=dict(family=p.family,load=p.load,plan_a=int(p.plan_a),plan_b=int(p.plan_b),T=p.T,X_a=p.X_a,X_b=p.X_b,F=p.F,R=p.R,C=p.C,
                 action_a=json.dumps(action(a)),action_b=json.dumps(action(b)),
                 fixed_action_state=int(f.name),fixed_action_feasible=bool(f.feasible),fixed_action_violation=float(f.violation),fixed_action_dominant_constraint=max(components,key=components.get),fixed_vmin=float(f.vmin),fixed_vmax=float(f.vmax),fixed_thermal=float(f.thermal),
                 loss_a_mw=float(a.loss),optimized_loss_b_mw=float(b.loss),fixed_action_loss_b_mw=float(f.loss),
                 true_delta_loss_mw=float(b.loss-a.loss),frozen_delta_loss_mw=float(f.loss-a.loss),delta_loss_error_mw=float(f.loss-b.loss),
                 capital_increment_million_cny=float(p.capital_increment),true_delta_objective_million_cny=float(exact_delta),frozen_delta_objective_million_cny=float(frozen_delta),delta_objective_error_million_cny=float(frozen_delta-exact_delta),
                 true_upgrade_economic=bool(exact_delta< -1e-7),frozen_upgrade_economic=bool(frozen_delta< -1e-7),economic_ranking_wrong=bool((exact_delta< -1e-7)!=(frozen_delta< -1e-7)),
                 frozen_shortcut_valid_physical_action=bool(f.feasible),independent_fixed_action_check_passed=True)
        rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(out/"switch_audit.csv",index=False)
    summary=dict(scope="All 11 S=0 adjacent-transformer pairs with both investments feasible and different exact optimal normal controls, across all eight frozen profiles. S=1 duplicates removed. No selective case omission.",comparison="Freeze small-investment exact optimal control at larger investment and run full AC; compare with large-investment exact control reoptimization",pairs=len(d),fixed_action_infeasible=int((~d.fixed_action_feasible).sum()),fixed_action_feasible_but_economic_ranking_wrong=int((d.fixed_action_feasible & d.economic_ranking_wrong).sum()),economic_ranking_wrong_total=int(d.economic_ranking_wrong.sum()),max_absolute_delta_loss_error_mw=float(d.delta_loss_error_mw.abs().max()),max_absolute_delta_objective_error_million_cny=float(d.delta_objective_error_million_cny.abs().max()),all_fixed_actions_independently_checked=True,corridor_outage_audit=corridor_outage_audit())
    (out/"switch_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(d[["family","load","plan_a","plan_b","fixed_action_feasible","true_delta_objective_million_cny","frozen_delta_objective_million_cny","economic_ranking_wrong"]].to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
