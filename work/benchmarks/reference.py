"""Finite investment/control reference enumeration for the new heterogeneous grids.

The AC engine is imported read-only from work/simulation/ac.py. Protocol-specific
network adjustments, if any, are explicit in NetworkPolicy and rebuilt before
solving. Nothing in this module changes ac.py or the original paper archive.
"""
from __future__ import annotations
from pathlib import Path
import sys, itertools, json, time, argparse, hashlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
import ac
import numpy as np
import pandas as pd

TAPS = (1.04, 1.075, 1.10)
CAPS = (0.0, 1.2)
SERVICE = (1.0, .95, .90, .80, .70, .60, .50, .40)
ANNUITY = sum(1.05**(-y) for y in (1, 2, 3))
LOSS_FACTOR = 8760 * ANNUITY * 600 / 1e6
UNSERVED_FACTOR = 8760 * ANNUITY * 10000 / 1e6
COST = dict(T=1.2, X=7/38*2, F=.9, S=.25, R=.15*2, C=.12*2)


def catalog():
    out = []
    for i, values in enumerate(itertools.product((0., .5, 1.), (0., .5, 1.), (0, 1), (0, 1), (0, 1), (0, 1))):
        z = dict(zip(("T", "X", "F", "S", "R", "C"), values))
        out.append(dict(plan=i, **z, capital=sum(COST[k]*z[k] for k in COST)))
    return out


def legal_ties():
    fc = ac.case("case33bw")
    ties = fc["branch"][fc["branch"][:,10] == 0]
    result = []
    for tie in range(len(ties)):
        es, active = ac.radial_edges(fc, fault=9, tie=tie)
        if len(active) == len(fc["bus"]) and len(es) == len(active)-1:
            result.append(tie)
    return tuple(result)


def controls(plan, event=0, service_grid=SERVICE):
    taps = TAPS if plan["R"] else (1.04,)
    caps = CAPS if plan["C"] else (0.,)
    ties = (-1,) if event not in (2, 3) else ((-1,) + legal_ties() if plan["S"] else (-1,))
    services = (1.,) if event == 0 else service_grid
    for service, tie, t0, t1, c0, c1 in itertools.product(services, ties, taps, taps, caps, caps):
        yield dict(taps=(t0,t1), caps=(c0,c1), service=service, tie=tie)


def build(family, load, plan, control, event=0, shape_seed=0, policy=None):
    """Default exactly preserves ac.py; modifications are declared by policy."""
    s = ac.build_network(family, load, plan=plan, control=control, event=event, shape_seed=shape_seed)
    if not policy or family != "mixed30":
        return s
    ppc = {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in s.ppc.items()}
    nmain = len(ac.case("case30")["branch"])
    ppc["branch"][:nmain,5:8] *= policy.get("main_rating_multiplier",1.)
    ppc["gen"][:,8] *= policy.get("pmax_multiplier",1.)
    for edge in policy.get("extra_corridor_edges",[]):
        ppc["branch"][edge,2:4] *= 1-.3*plan["T"]
        ppc["branch"][edge,5:8] *= 1+.5*plan["T"]
    meta = dict(s.meta, explicit_policy=policy)
    return ac.ACSystem(ppc, meta)


def evaluate(family,load,plan,control,event=0,shape_seed=0,policy=None):
    start=time.perf_counter()
    s=build(family,load,plan,control,event,shape_seed,policy)
    r=s.solve(tol=1e-10,maxiter=30)
    m=s.metrics(r["x"])
    violation=max((.9-m["vmin"])/.1,(m["vmax"]-1.1)/.1,m["thermal"]-1,m["genviol"])
    feasible=bool(r["converged"] and violation<=1e-7)
    return dict(family=family,load=load,shape_seed=shape_seed,event=event,
                T=plan["T"],X=plan["X"],F=plan["F"],tap0=control["taps"][0],tap1=control["taps"][1],
                cap0=control["caps"][0],cap1=control["caps"][1],service=control["service"],tie=control["tie"],
                nb=s.nb,nvar=s.n,requested_mw=s.meta["requested_mw"],served_mw=s.meta["served_mw"],
                unserved_mw=s.meta["unserved_mw"],converged=r["converged"],feasible=feasible,
                violation=violation,residual=r["residual"],iterations=r["iterations"],**m,seconds=time.perf_counter()-start)


def normal_reference(family,load,shape_seed=0,policy=None):
    """Evaluate each distinct physical state once, then recover all 144 menus.

    R/C/S determine menu membership; they do not independently change the power
    flow for a given tap/cap/topology setting. Deduplication preserves exactness.
    """
    started=time.perf_counter();states=[]
    maxmenu=dict(T=0.,X=0.,F=0,S=1,R=1,C=1)
    for t,x,f in itertools.product((0.,.5,1.),(0.,.5,1.),(0,1)):
        plan=dict(maxmenu,T=t,X=x,F=f)
        for control in controls(plan):
            states.append(evaluate(family,load,plan,control,shape_seed=shape_seed,policy=policy))
    states=pd.DataFrame(states);plans=[]
    for p in catalog():
        mask=states["T"]==p["T"]
        mask &= (states.X==p["X"]) & (states.F==p["F"])
        if not p["R"]:mask &= (states.tap0==1.04) & (states.tap1==1.04)
        if not p["C"]:mask &= (states.cap0==0.) & (states.cap1==0.)
        allrows=states[mask];good=allrows[allrows.feasible]
        if len(good):
            best=good.sort_values(["loss","tap0","tap1","cap0","cap1"]).iloc[0]
            plans.append(dict(family=family,load=load,shape_seed=shape_seed,**p,feasible=True,controls=len(allrows),feasible_controls=len(good),objective=p["capital"]+LOSS_FACTOR*best.loss,loss=best.loss,selected_state=int(best.name),tap0=best.tap0,tap1=best.tap1,cap0=best.cap0,cap1=best.cap1,vmin=best.vmin,vmax=best.vmax,thermal=best.thermal,genviol=best.genviol))
        else:
            plans.append(dict(family=family,load=load,shape_seed=shape_seed,**p,feasible=False,controls=len(allrows),feasible_controls=0,objective=np.nan,loss=np.nan,selected_state=-1,min_violation=allrows.violation.min()))
    plans=pd.DataFrame(plans)
    summary=dict(family=family,load=load,shape_seed=shape_seed,distinct_ac_states=len(states),feasible_ac_states=int(states.feasible.sum()),feasible_investments=int(plans.feasible.sum()),seconds=time.perf_counter()-started,policy=policy or {})
    if plans.feasible.any():
        win=plans[plans.feasible].sort_values(["objective","plan"]).iloc[0]
        summary.update(best_plan=int(win.plan),best_objective=float(win.objective),best_capital=float(win.capital),best_T=float(win["T"]),best_X=float(win.X),best_F=int(win.F),best_R=int(win.R),best_C=int(win.C),best_taps=[float(win.tap0),float(win.tap1)],best_caps=[float(win.cap0),float(win.cap1)])
    return states,plans,summary


def recourse_reference(family,load,plan,event,shape_seed=0,service_grid=SERVICE):
    """Exact service-first/loss-second reference within the declared menu.

    Every tap/capacitor control at the highest feasible served-demand level is
    assessed. Lower served levels need not be solved once that level is known.
    Islanded and fully restored topologies retain their own active-bus dimensions.
    """
    started=time.perf_counter();groups={}
    for control in controls(plan,event,service_grid):
        key=(control["service"],control["tie"])
        groups.setdefault(key,[]).append(control)
    ordered=[]
    for key,group in groups.items():
        system=build(family,load,plan,group[0],event,shape_seed)
        ordered.append((float(system.meta["served_mw"]),key,group))
    ordered.sort(key=lambda z:(-z[0],z[1]))
    states=[];best=None
    for served,key,group in ordered:
        if best is not None and served<best["served_mw"]-1e-8:
            break
        for control in group:
            state=evaluate(family,load,plan,control,event,shape_seed)
            state["state_index"]=len(states);states.append(state)
            if state["feasible"] and (best is None or state["loss"]<best["loss"]-1e-12):
                best=state
    summary=dict(family=family,load=load,shape_seed=shape_seed,event=event,plan=int(plan["plan"]),
                 **{k:plan[k] for k in COST},feasible=best is not None,
                 distinct_ac_states=len(states),seconds=time.perf_counter()-started)
    if best is not None:
        summary.update({k:best[k] for k in ["state_index","tap0","tap1","cap0","cap1","service","tie","nb","nvar","requested_mw","served_mw","unserved_mw","loss","vmin","vmax","thermal","genviol"]})
    return pd.DataFrame(states),summary


def run_pilot(policy=None,label="unmodified"):
    out=HERE/"pilot"/label;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(status="Pilot protocol; outcomes determine the final declared benchmark, not claimed as unseen validation",family_loads={"mixed9":[.90,1.10,1.25,1.40],"mixed30":[.60,.70,.80,.90]},shape_seed=20261108,voltage_limits=[.9,1.1],taps=list(TAPS),capacitors=list(CAPS),costs=COST,normal_full_supply=True,policy=policy or {},ac_source_sha256=hashlib.sha256((HERE.parent/"simulation/ac.py").read_bytes()).hexdigest())
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    summaries=[]
    for fam,loads in protocol["family_loads"].items():
        for load in loads:
            states,plans,summary=normal_reference(fam,load,protocol["shape_seed"],policy)
            stem=f"{fam}_{load:.2f}"
            states.to_csv(out/f"{stem}_states.csv",index=False);plans.to_csv(out/f"{stem}_plans.csv",index=False)
            summaries.append(summary);print(json.dumps(summary),flush=True)
            (out/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    return summaries


if __name__ == "__main__":
    run_pilot()
