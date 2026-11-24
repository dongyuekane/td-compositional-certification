"""Derive comparable investment pairs and small, freshly solved fault references."""
from reference import *
from pypower.api import runpf, ppoption


def pairs():
    root=HERE/"pilot/unmodified";rows=[]
    for file in sorted(root.glob("*_plans.csv")):
        p=pd.read_csv(file)
        for a in p.itertuples():
            if a.X>=1:continue
            b=p[(p["T"]==a.T)&(p.X==a.X+.5)&(p.F==a.F)&(p.S==a.S)&(p.R==a.R)&(p.C==a.C)].iloc[0]
            row=dict(family=a.family,load=a.load,shape_seed=a.shape_seed,plan_a=int(a.plan),plan_b=int(b.plan),feasible_a=bool(a.feasible),feasible_b=bool(b.feasible),X_a=a.X,X_b=b.X,T=a.T,F=a.F,S=a.S,R=a.R,C=a.C,capital_increment=b.capital-a.capital)
            if a.feasible and b.feasible:
                row.update(loss_a=a.loss,loss_b=b.loss,loss_difference=b.loss-a.loss,objective_difference=b.objective-a.objective,
                           state_a=int(a.selected_state),state_b=int(b.selected_state),
                           tap_switch=bool(a.tap0!=b.tap0 or a.tap1!=b.tap1),cap_switch=bool(a.cap0!=b.cap0 or a.cap1!=b.cap1))
                for key in ["tap0","tap1","cap0","cap1"]:row[key+"_a"]=getattr(a,key);row[key+"_b"]=b[key]
            rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(root/"transformer_increment_pairs.csv",index=False)
    good=result[result.feasible_a&result.feasible_b]
    for (fam,ld),g in good.groupby(["family","load"]):
        print("PAIRS",fam,ld,"both feasible",len(g),"tap/cap switch",int((g.tap_switch|g.cap_switch).sum()),"worthwhile",int((g.objective_difference<0).sum()),flush=True)


def faults():
    out=HERE/"pilot/faults";out.mkdir(parents=True,exist_ok=True)
    cases=[("mixed9",1.1),("mixed9",1.25),("mixed30",.7),("mixed30",.8)]
    protocol=dict(status="pilot exact finite-menu fault references",cases=cases,plans=[31,47],events={1:"transmission branch outage specified by ac.py",2:"case33bw normal branch index 9 removed; legal single ties available when S=1"},legal_tie_indices=list(legal_ties()),service_grid=list(SERVICE),shape_seed=20261108,normal_voltage_limits=[.9,1.1],postfault_voltage_limits=[.9,1.1],operating_policy="maximize served active demand; minimize active loss at that service level",topology_note="Unrestored states have fewer active variables. Pairwise certificate construction must match active-bus sets or use separately proved bounds.")
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    cc={c["plan"]:c for c in catalog()};summaries=[];checks=[]
    for family,load in cases:
        for pid,event in itertools.product([31,47],[1,2]):
            plan=cc[pid];states,summary=recourse_reference(family,load,plan,event,20261108)
            stem=f"{family}_{load:.2f}_p{pid}_e{event}"
            states.to_csv(out/f"{stem}_states.csv",index=False)
            if summary["feasible"]:
                c=dict(taps=(summary["tap0"],summary["tap1"]),caps=(summary["cap0"],summary["cap1"]),service=summary["service"],tie=int(summary["tie"]))
                s=build(family,load,plan,c,event,20261108);r=s.solve();reference,ok=runpf(s.ppc,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-11))
                v=reference["bus"][:,7]*np.exp(1j*reference["bus"][:,8]*np.pi/180)
                voltage_error=float(max(abs(r["voltage"]-v)));loss=float(reference["gen"][:,1].sum()-reference["bus"][:,2].sum());loss_error=abs(loss-r["loss"])
                assert ok and voltage_error<1e-7 and loss_error<1e-7
                checks.append(dict(family=family,load=load,plan=pid,event=event,voltage_error_pu=voltage_error,loss_error_mw=loss_error,passed=True))
            summaries.append(summary);print("FAULT",json.dumps(summary),flush=True)
            (out/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
            pd.DataFrame(checks).to_csv(out/"independent_pypower_checks.csv",index=False)


if __name__=="__main__":
    pairs();faults()
