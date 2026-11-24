"""Freeze pilot-derived profiles, common-investment references and Newton traces."""
from reference import *
import datetime


def capture(csv_path):
    frame=pd.read_csv(csv_path);maxn=int(frame.nvar.max())
    checkpoints=np.full((len(frame),4,maxn),np.nan);lengths=[];residuals=[];active_keys=[]
    start=time.perf_counter()
    for i,a in enumerate(frame.itertuples()):
        plan=dict(T=a.T,X=a.X,F=int(a.F),S=1,R=1,C=1)
        c=dict(taps=(a.tap0,a.tap1),caps=(a.cap0,a.cap1),service=a.service,tie=int(a.tie))
        s=build(a.family,a.load,plan,c,int(a.event),int(a.shape_seed));r=s.solve(tol=1e-10,maxiter=30,trace=True)
        assert r["converged"]==bool(a.converged)
        assert abs(r["loss"]-a.loss)<1e-8
        for j,k in enumerate((1,2,3,-1)):
            checkpoints[i,j,:s.n]=r["trace"][k if k<0 else min(k,len(r["trace"])-1)]
        lengths.append(r["iterations"]);residuals.append(r["residual"])
        active_keys.append(json.dumps([f["active"] for f in s.meta["feeders"]],separators=(",",":")))
    target=csv_path.with_name(csv_path.stem.replace("_states","_traces")+".npz")
    np.savez_compressed(target,x=checkpoints,steps=np.array([1,2,3,-1]),row_index=np.arange(len(frame)),nvar=frame.nvar.to_numpy(),iterations=np.array(lengths),full_residual=np.array(residuals),active_bus_sets=np.array(active_keys))
    entry=dict(csv=str(csv_path.relative_to(HERE)),npz=str(target.relative_to(HERE)),rows=len(frame),seconds=time.perf_counter()-start,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    print("TRACES",entry,flush=True)
    return entry


def common_investments():
    out=HERE/"frozen_v1";out.mkdir(exist_ok=True)
    summary={}
    for family,loads,stress in [("mixed9",[.9,1.1,1.25],1.4),("mixed30",[.6,.7,.8],.9)]:
        frames=[pd.read_csv(HERE/f"pilot/unmodified/{family}_{load:.2f}_plans.csv") for load in loads]
        allp=pd.concat(frames,ignore_index=True)
        grouped=allp.groupby("plan").agg(feasible=("feasible","all"),finite_profile_count=("objective","count"),objective=("objective","mean"),capital=("capital","first"),T=("T","first"),X=("X","first"),F=("F","first"),S=("S","first"),R=("R","first"),C=("C","first"))
        grouped.loc[~grouped.feasible,"objective"]=np.nan
        assert (grouped[grouped.feasible].finite_profile_count==3).all()
        grouped.to_csv(out/f"{family}_normal_common_portfolios.csv")
        win=grouped[grouped.feasible].sort_values(["objective","capital"]).iloc[0]
        hold=pd.read_csv(HERE/f"pilot/unmodified/{family}_{stress:.2f}_plans.csv").set_index("plan")
        h=hold.loc[win.name]
        summary[family]=dict(development_loads=loads,equal_weights=[1/3]*3,stress_load=stress,catalog_size=144,common_feasible_portfolios=int(grouped.feasible.sum()),selected_plan=int(win.name),selected_objective=float(win.objective),selected_capital=float(win.capital),selected_investment={k:float(win[k]) for k in COST},stress_selected_feasible=bool(h.feasible),stress_catalog_feasible=int(hold.feasible.sum()),stress_selected_objective=float(h.objective) if h.feasible else None)
    (out/"normal_common_planning_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("COMMON INVESTMENT",json.dumps(summary),flush=True)
    return summary


def main():
    out=HERE/"frozen_v1";out.mkdir(exist_ok=True)
    protocol=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status="Declared synthetic benchmark frozen after feasibility pilot; development and stress cases are not independent field validation",family_loads={"mixed9":{"development":[.9,1.1,1.25],"stress":[1.4]},"mixed30":{"development":[.6,.7,.8],"stress":[.9]}},shape_seed=20261108,equal_development_weights=True,voltage_limits_pu=[.9,1.1],limits_policy="Original case30 branch and generator ratings retained; no headroom adjustment",planning_scope="One common investment per family across the three normal development profiles. Fault references are separate finite-recourse diagnostics for selected portfolios, not a complete fault-constrained investment enumeration.",investment_levels={"T":[0,.5,1],"X":[0,.5,1],"F":[0,1],"S":[0,1],"R":[0,1],"C":[0,1]},costs_million_cny=COST,cost_status="Explicit simulation assumptions, not vendor or utility quotations. T is scaled by the 100-to-10 MVA transmission model scaling; X/R/C are repriced for two physical interfaces; F/S retain target-feeder assumptions.",budget=None,objective="Capital plus three-year discounted annual normal loss cost; equally weighted profiles; 600 CNY/MWh, 8760 hours/year, discount 0.05",taps=list(TAPS),caps_mvar=list(CAPS),fault_service_grid=list(SERVICE),precision=dict(newton_residual=1e-10,feasibility_tolerance=1e-7),checkpoint_steps=[1,2,3,"converged"],checkpoint_start="Flat Cartesian voltage; no warm start",source_sha256=hashlib.sha256((HERE.parent/"simulation/ac.py").read_bytes()).hexdigest())
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    summary=common_investments()
    files=sorted((HERE/"pilot/unmodified").glob("*_states.csv"))+sorted((HERE/"pilot/faults").glob("*_states.csv"))
    traces=[]
    for file in files:
        traces.append(capture(file));(out/"trace_manifest.json").write_text(json.dumps(traces,indent=2)+"\n")
    checks=pd.read_csv(HERE/"pilot/faults/independent_pypower_checks.csv")
    final=dict(common_planning=summary,normal_reference_states=sum(x["rows"] for x in traces if "/unmodified/" in x["csv"]),fault_reference_states=sum(x["rows"] for x in traces if "/faults/" in x["csv"]),trace_bytes=sum(x["bytes"] for x in traces),trace_generation_seconds=sum(x["seconds"] for x in traces),independently_checked_fault_optima=len(checks),max_fault_voltage_error_pu=float(checks.voltage_error_pu.max()),max_fault_loss_error_mw=float(checks.loss_error_mw.max()))
    (out/"summary.json").write_text(json.dumps(final,indent=2)+"\n")


if __name__=="__main__":main()
