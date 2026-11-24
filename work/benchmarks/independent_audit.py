"""Stratified independent PYPOWER checks of frozen normal and fault references."""
from reference import *
from pypower.api import runpf, ppoption


def independent_metrics(result):
    """Read PYPOWER voltages, terminal flows and generator outputs directly."""
    bus,br,gen=result["bus"],result["branch"],result["gen"]
    active=(br[:,10]>0)&(br[:,5]>0)
    sfrom=np.hypot(br[:,13],br[:,14]);sto=np.hypot(br[:,15],br[:,16])
    thermal=float(np.max(np.maximum(sfrom[active],sto[active])/br[active,5],initial=0.))
    gv=np.r_[(gen[:,1]-gen[:,8])/np.maximum(1.,gen[:,8]),(gen[:,9]-gen[:,1])/np.maximum(1.,gen[:,8]),(gen[:,2]-gen[:,3])/np.maximum(1.,abs(gen[:,3])),(gen[:,4]-gen[:,2])/np.maximum(1.,abs(gen[:,4]))]
    margin=max((.9-float(bus[:,7].min()))/.1,(float(bus[:,7].max())-1.1)/.1,thermal-1,float(gv.max()))
    return dict(thermal=thermal,genviol=float(gv.max()),violation=margin,loss=float(gen[:,1].sum()-bus[:,2].sum()),safe=bool(margin<=1e-7))


def selected_rows():
    selected={}
    def add(file,index,reason):
        key=(str(file),int(index));selected.setdefault(key,set()).add(reason)
    for file in sorted((HERE/"pilot/unmodified").glob("*_states.csv")):
        d=pd.read_csv(file)
        for _,g in d.groupby(["T","X","F"]):
            add(file,(g.violation.abs()).idxmin(),"nearest_boundary_in_T_X_F_stratum")
        for feasible in [False,True]:
            g=d[d.feasible==feasible]
            if len(g):
                for i in g.violation.abs().nlargest(2).index:add(file,i,"interior_feasible" if feasible else "clearly_infeasible")
    pairs=pd.read_csv(HERE/"pilot/unmodified/transformer_increment_pairs.csv")
    for a in pairs[(pairs.tap_switch==True)|(pairs.cap_switch==True)].itertuples():
        file=HERE/f"pilot/unmodified/{a.family}_{a.load:.2f}_states.csv"
        add(file,a.state_a,"investment_control_switch_a");add(file,a.state_b,"investment_control_switch_b")
    fault_summary=json.loads((HERE/"pilot/faults/summary.json").read_text())
    for a in fault_summary:
        file=HERE/f"pilot/faults/{a['family']}_{a['load']:.2f}_p{a['plan']}_e{a['event']}_states.csv"
        d=pd.read_csv(file)
        for feasible in [False,True]:
            g=d[d.feasible==feasible]
            if len(g):
                add(file,g.violation.abs().idxmin(),"fault_near_boundary")
                add(file,g.violation.abs().idxmax(),"fault_interior")
        if a["feasible"]:add(file,a["state_index"],"fault_selected_control")
    return selected


def main():
    started=time.perf_counter();chosen=selected_rows();rows=[];cache={}
    for (file,index),reasons in chosen.items():
        path=Path(file)
        if file not in cache:
            traces=np.load(path.with_name(path.stem.replace("_states","_traces")+".npz"))
            cache[file]=(pd.read_csv(path),traces["x"],traces["nvar"])
        data,x,nvar=cache[file];a=data.loc[index]
        plan=dict(T=a["T"],X=a.X,F=int(a.F),S=1,R=1,C=1)
        control=dict(taps=(a.tap0,a.tap1),caps=(a.cap0,a.cap1),service=a.service,tie=int(a.tie))
        s=build(a.family,a.load,plan,control,int(a.event),int(a.shape_seed))
        result,ok=runpf(s.ppc,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-11))
        assert ok==bool(a.converged),(file,index,ok,a.converged)
        full=x[index,-1,:int(nvar[index])];v=s.voltage(full)
        vp=result["bus"][:,7]*np.exp(1j*result["bus"][:,8]*np.pi/180)
        m=independent_metrics(result)
        verr=float(max(abs(v-vp)));lerr=abs(float(a.loss)-m["loss"]);merr=abs(float(a.violation)-m["violation"])
        agree=bool(a.feasible)==m["safe"]
        assert verr<1e-7 and lerr<1e-7 and merr<1e-6 and agree,(file,index,verr,lerr,merr,agree)
        rows.append(dict(file=str(path.relative_to(HERE)),row=index,reasons=";".join(sorted(reasons)),family=a.family,load=a.load,event=int(a.event),T=a["T"],X=a.X,F=int(a.F),tap0=a.tap0,tap1=a.tap1,cap0=a.cap0,cap1=a.cap1,service=a.service,tie=int(a.tie),reference_feasible=bool(a.feasible),independent_feasible=m["safe"],feasibility_agrees=agree,voltage_error_pu=verr,loss_error_mw=lerr,margin_error=merr,reference_margin=a.violation,independent_margin=m["violation"]))
    d=pd.DataFrame(rows);out=HERE/"frozen_v1";d.to_csv(out/"stratified_independent_pypower_checks.csv",index=False)
    uniquephysical=d[["T","X","F"]].drop_duplicates()
    assert len(d)>=100 and len(uniquephysical)==18
    summary=dict(sample_selection="Deterministic stratification: closest security boundary for all T/X/F strata in every normal profile; interior feasible/infeasible cases; every unique normal selected-control switch; fault boundaries/interiors and optima",checked_states=len(d),normal_checked=int((d.event==0).sum()),fault_checked=int((d.event>0).sum()),feasible_checked=int(d.reference_feasible.sum()),infeasible_checked=int((~d.reference_feasible).sum()),control_switch_checked=int(d.reasons.str.contains("investment_control_switch").sum()),physical_T_X_F_strata=len(uniquephysical),families=sorted(d.family.unique().tolist()),profiles=int(d[["family","load"]].drop_duplicates().shape[0]),max_voltage_error_pu=float(d.voltage_error_pu.max()),max_loss_error_mw=float(d.loss_error_mw.max()),max_margin_error=float(d.margin_error.max()),feasibility_agreement_count=int(d.feasibility_agrees.sum()),feasibility_agreement_rate=float(d.feasibility_agrees.mean()),seconds=time.perf_counter()-started)
    (out/"stratified_independent_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    dataset_summary=out/"summary.json"
    if dataset_summary.exists():
        aggregate=json.loads(dataset_summary.read_text());aggregate["stratified_independent_audit"]=summary
        dataset_summary.write_text(json.dumps(aggregate,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
