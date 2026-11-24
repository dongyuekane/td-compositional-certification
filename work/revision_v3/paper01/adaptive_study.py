"""Decision-targeted refinement with reusable component certificates.

No reference feasibility, costs, or selected plans enter the algorithm.
Coarse and refined enclosures concern the same labelled high-voltage branch.
The implementation inherits the original guarded-binary64 certificate scope.
"""
from pathlib import Path
import sys, time, json, itertools, argparse
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SRC=ROOT/'research/normalized_response'
sys.path[:0]=[str(SRC),str(ROOT/'research/control_blocks')]
from run_study import (Atlas, baseline, ac, NetworkReuse, MainTemplates, FastReduced,
                       snake, FAMILIES, TAPS, CAPS, catalog, LOSS_FACTOR)
from atlas import Response, feeder_at, power_conversion, ch, PPoly
from feeder_tube import certify_response
from coupled_certificate import certify_main, classify, slack_loss_interval


class Experiment:
    def __init__(self,family,mode,coarse=3,capital_scale=1.):
        self.family=family; self.mode=mode; self.coarse=coarse;self.capital_scale=capital_scale;self.started=time.perf_counter()
        self.atlas=Atlas(coarse if mode=='adaptive' else 6,domain=(.4,1.8))
        self.tubes={k:certify_response(r) for k,r in self.atlas.responses.items()}
        assert all(t['certified'] for t in self.tubes.values())
        self.degrees={k:coarse if mode=='adaptive' else 6 for k in self.tubes}
        if mode=='direct_output':
            for k,tube in self.tubes.items():
                response=self.atlas.responses[k]
                for name,mat in [('current_from',response.system.Yf),('current_to',response.system.Yt)]:
                    tube[name]['error']=abs(mat[:,1:])@np.array(tube['voltage_error'])+2e-12
                    tube[name]['derivative_bound']=abs(mat[:,1:])@np.array(tube['voltage_derivative_bound'])+2e-12
        self.offline=time.perf_counter()-self.started
        self.samples=sum(r.samples for r in self.atlas.responses.values())
        self.refined=[]; self.updates=0; self.records=[]; self.models=[];self.solutions=[]
        self.component_nesting=[];self.main_nesting=[]
        self.lkeys=[];self.versions=[];self.meta=[]
        self.plans=[p for p in catalog() if not p['S']]
        self.cap=capital_scale*np.array([p['capital'] for p in self.plans])
        self.groups=[]
        self.numerical=0.; self.selection=0.; self.trace=[]
        ac.case=baseline.CACHED_CASE
        for j,load in enumerate(FAMILIES[family]):
            maker=NetworkReuse(family,load);templates=MainTemplates();last=None
            for gi,(T,X,F) in enumerate(snake(((0.,.5,1.),(0.,.5,1.),(0,1)))):
                p=dict(T=T,X=X,F=F,S=1,R=1,C=1)
                controls=snake((TAPS,TAPS,CAPS,CAPS))
                for ta,tb,ca,cb in (controls if gi%2==0 else controls[::-1]):
                    c=dict(taps=(ta,tb),caps=(ca,cb),service=1.,tie=-1)
                    s=maker.system(p,c);model=FastReduced(s,self.atlas,templates)
                    starts=[model.main.flat()]
                    if last is not None:starts.append(last)
                    x0=min(starts,key=lambda x:float(max(abs(model.residual(x)))))
                    r=model.solve(x0);last=r['main_x'].copy()
                    keys=[(k,X,F if k==0 else 0,c['taps'][k],c['caps'][k]) for k in (0,1)]
                    self.models.append(model);self.solutions.append(r);self.lkeys.append(keys)
                    self.versions.append(tuple(self.degrees[k] for k in keys))
                    self.meta.append(dict(profile=j,load=load,T=T,X=X,F=F,tap0=ta,tap1=tb,cap0=ca,cap1=cb))
                    self.records.append(self.certify(model,r,keys))
        self.initial_seconds=time.perf_counter()-self.started
        self.lo=np.array([r['lo'] for r in self.records]);self.hi=np.array([r['hi'] for r in self.records])
        self.status=np.array([r['status'] for r in self.records])
        self.predloss=np.array([r['prediction'] for r in self.records])
        self.predviol=np.array([r['predviol'] for r in self.records])
        meta=pd.DataFrame(self.meta)
        for p in self.plans:
            shared=(meta['T']==p['T'])&(meta.X==p['X'])&(meta.F==p['F'])
            if not p['R']:shared&=(meta.tap0==1.04)&(meta.tap1==1.04)
            if not p['C']:shared&=(meta.cap0==0)&(meta.cap1==0)
            self.groups.append([np.flatnonzero(shared&(meta.profile==j)) for j in range(3)])
        # Dense small menus avoid repeated DataFrame construction in refinement.
        self.menu=np.zeros((len(self.plans),3,36),dtype=int)
        self.mask=np.zeros_like(self.menu,dtype=bool)
        for pi,gs in enumerate(self.groups):
            for j,g in enumerate(gs):self.menu[pi,j,:len(g)]=g;self.mask[pi,j,:len(g)]=True

    def certify(self,model,r,keys):
        ts=[self.tubes[k] for k in keys]
        cert=certify_main(model,r,ts);cl=classify(model,r,cert,ts)
        lo,hi=slack_loss_interval(model,r,cert)
        # This passive benchmark has nonnegative line resistance and shunt conductance.
        # Zero is therefore an independent lower bound, including failed inclusions.
        assert (model.full.ppc['branch'][:,2]>=0).all() and (model.full.ppc['bus'][:,4]>=0).all()
        metrics=model.full.metrics(r['x'])
        pv=max((.9-metrics['vmin'])/.1,(metrics['vmax']-1.1)/.1,metrics['thermal']-1,metrics['genviol'])
        return dict(lo=max(0.,lo),hi=hi,status=cl['status'],prediction=metrics['loss'],predviol=pv,
                    violation_lower=cl.get('violation_lower',-np.inf),
                    violation_upper=cl.get('violation_upper',np.inf),certified=cert['certified'],
                    radius=cert.get('radius',np.inf),error=cert.get('error',np.inf))

    def bounds(self):
        valid=self.mask & (self.status[self.menu]!='violating')
        feasible=self.mask & (self.status[self.menu]=='feasible')
        los=np.where(valid,self.lo[self.menu],np.inf)
        his=np.where(feasible,self.hi[self.menu],np.inf)
        gl=los.min(axis=2);gu=his.min(axis=2)
        return self.cap+LOSS_FACTOR*gl.mean(axis=1),self.cap+LOSS_FACTOR*gu.mean(axis=1),gl,gu,los,his

    def update(self,i):
        t=time.perf_counter(); keys=self.lkeys[i]
        for k in keys:
            if self.degrees[k]!=6:
                oldr=self.atlas.responses[k]
                if self.degrees[k]==3:
                    r=Response.__new__(Response);r.system=oldr.system;r.domain=oldr.domain
                    nodes=np.cos(np.pi*np.arange(7)/6); zs=.5*(sum(r.domain)+(r.domain[1]-r.domain[0])*nodes)
                    values=ch.chebval(nodes,oldr.coeff).T
                    original=r.system.spec.copy();iters=0
                    for idx in [5,3,1]:
                        r.system.spec=original*zs[idx]
                        zsol=r.system.solve(maxiter=30)
                        assert zsol['converged'];v=zsol['voltage'];power=v[0]*np.conj(r.system.Y[0]@v)
                        values[idx]=np.r_[power.real,power.imag,v[1:].real,v[1:].imag];iters+=zsol['iterations']
                    r.system.spec=original
                    r.coeff=ch.chebfit(nodes,values,6)
                    r.dcoeff=ch.chebder(r.coeff,axis=0)*2/(r.domain[1]-r.domain[0])
                    shifted=power_conversion(6,r.domain[1]-r.domain[0])@r.coeff
                    r.polynomial=PPoly(shifted[::-1,None,:],np.array(r.domain));r.derivative=r.polynomial.derivative()
                    r.iterations=iters;r.samples=3
                else:r=Response(feeder_at(*k),6,domain=(.4,1.8))
                newtube=certify_response(r);assert newtube['certified']
                diff=r.coeff.copy();diff[:len(oldr.coeff)]-=oldr.coeff;m=r.system.m
                delta=np.sum(abs(diff[:,2:2+m]+1j*diff[:,2+m:]),axis=0)
                nesting=float(max((delta+np.array(newtube['voltage_error']))/np.array(self.tubes[k]['voltage_error'])))
                assert nesting<=1, 'Refined tube must enclose the same component branch'
                self.component_nesting.append(nesting)
                self.atlas.responses[k]=r;self.tubes[k]=newtube
                self.degrees[k]=6;self.samples+=r.samples;self.refined.append(k)
        model=self.models[i]
        model.responses=[self.atlas.responses[k] for k in keys]
        r=model.solve(self.solutions[i]['main_x']);new=self.certify(model,r,keys)
        old=self.records[i]
        if old['certified'] and new['certified']:
            nesting=(float(max(abs(r['main_x']-self.solutions[i]['main_x'])))+new['error'])/old['radius']
            assert nesting<=1, 'Refined operating root must lie in the previous inclusion neighborhood'
            self.main_nesting.append(nesting)
        assert not (old['status']=='feasible' and new['status']=='violating')
        assert not (old['status']=='violating' and new['status']=='feasible')
        self.lo[i]=max(self.lo[i],new['lo']);self.hi[i]=min(self.hi[i],new['hi'])
        if new['status']!='unresolved':self.status[i]=new['status']
        assert self.lo[i]<=self.hi[i]+1e-9
        self.records[i]=dict(new,lo=self.lo[i],hi=self.hi[i],status=self.status[i])
        self.solutions[i]=r;self.versions[i]=(6,6);self.updates+=1
        self.numerical+=time.perf_counter()-t

    def run(self):
        for it in range(1945):
            t=time.perf_counter();L,U,gl,gu,los,his=self.bounds()
            best=int(np.argmin(U)); rival=L.copy();rival[best]=np.inf
            gap=rival.min()-U[best]
            if np.isfinite(U[best]) and gap>0:break
            if self.mode!='adaptive':break
            # Candidate portfolios whose lower bounds can still beat the incumbent.
            eligible=np.flatnonzero((L<=U[best])&np.isfinite(L))
            if not len(eligible):raise RuntimeError('No eligible investment')
            # First obtain a feasible incumbent using unverified values only as priorities.
            if not np.isfinite(U[best]):
                prediction=[]
                for pi in eligible:
                    vals=[]
                    for g in self.groups[pi]:
                        q=g[self.predviol[g]<=1e-7]
                        vals.append(self.predloss[q].min() if len(q) else np.inf)
                    prediction.append(self.cap[pi]+LOSS_FACTOR*np.mean(vals))
                pi=int(eligible[np.argmin(prediction)])
                js=np.flatnonzero(~np.isfinite(gu[pi]));j=int(js[0])
                g=self.groups[pi][j]
                cand=g[(self.status[g]!='violating')&np.array([self.versions[i]!=(6,6) for i in g])]
                if not len(cand):raise RuntimeError('Unresolved finest witness')
                i=int(min(cand,key=lambda x:(self.predviol[x]>1e-7,self.predloss[x])))
            else:
                contenders=eligible[eligible!=best]
                pi=int(contenders[np.argmin(L[contenders])]) if len(contenders) else best
                # A lower-bound minimizer blocks exclusion; resolve that action.
                blockers=[]
                for j,g in enumerate(self.groups[pi]):
                    possible=g[self.status[g]!='violating']
                    if not len(possible):continue
                    low=self.lo[possible].min()
                    cand=possible[(self.lo[possible]<=low+1e-12)&np.array([self.versions[i]!=(6,6) for i in possible])]
                    if len(cand):
                        i0=int(min(cand,key=lambda x:self.predloss[x]))
                        width=gu[pi,j]-gl[pi,j]
                        blockers.append((width,j,i0))
                if not blockers:
                    # An already-tight incumbent may need a better feasible witness.
                    for j,g in enumerate(self.groups[pi]):
                        cand=g[(self.status[g]!='violating')&np.array([self.versions[i]!=(6,6) for i in g])&(self.lo[g]<gu[pi,j])]
                        if len(cand):blockers.append((gu[pi,j]-self.lo[cand].min(),j,int(cand[np.argmin(self.lo[cand])])))
                if not blockers:
                    pi=best
                    for j,g in enumerate(self.groups[pi]):
                        cand=g[(self.status[g]=='feasible')&np.array([self.versions[i]!=(6,6) for i in g])]
                        if len(cand):
                            i0=int(cand[np.argmin(self.hi[cand])])
                            blockers.append((self.hi[i0]-self.lo[i0],j,i0))
                if not blockers:raise RuntimeError(f'Finest bounds overlap: decision unresolved; best={best}, gap={gap}, updates={self.updates}')
                _,j,i=max(blockers,key=lambda b:(b[0],-b[1]))
            self.selection+=time.perf_counter()-t
            self.trace.append(dict(iteration=it,plan=self.plans[pi]['plan'],profile=j,state=i,
                                   lower=float(L[pi]),incumbent_upper=float(U[best]),labels_refined=len(self.refined)))
            self.update(i)
        else:raise RuntimeError('Refinement exhausted')
        total=time.perf_counter()-self.started
        sel=self.plans[best]['plan']
        result=dict(family=self.family,mode=self.mode,coarse=self.coarse,capital_scale=self.capital_scale,selected_plan=sel,lower=float(L[best]),upper=float(U[best]),
                    gap=float(gap),total_seconds=total,initial_seconds=self.initial_seconds,
                    refinement_seconds=self.numerical,selection_seconds=self.selection,
                    refined_labels=len(self.refined),labels_total=len(self.tubes),training_samples=self.samples,
                    refined_states=self.updates,states=len(self.models),unresolved=int((self.status=='unresolved').sum()),
                    offline_seconds=self.offline,max_component_nesting=max(self.component_nesting,default=0.),
                    max_main_nesting=max(self.main_nesting,default=0.))
        # Independent frozen references are read only after the timed decision.
        audits=[]
        d=pd.DataFrame(self.meta);d['lo']=self.lo;d['hi']=self.hi;d['status']=self.status
        for load in FAMILIES[self.family]:
            ref=pd.read_csv(ROOT/f'benchmarks/pilot/unmodified/{self.family}_{load:.2f}_states.csv')
            keys=['T','X','F','tap0','tap1','cap0','cap1']
            a=d[d.load==load].merge(ref,on=keys,validate='one_to_one')
            assert ((a.loss>=a.lo-1e-8)&(a.loss<=a.hi+1e-8)).all()
            assert not ((a.status=='feasible')&~a.feasible).any()
            assert not ((a.status=='violating')&a.feasible).any()
            audits.append(a)
        truth=pd.read_csv(ROOT/f'benchmarks/frozen_v1/{self.family}_normal_common_portfolios.csv')
        capital={p['plan']:p['capital'] for p in catalog()}
        truth['objective']+=(self.capital_scale-1)*truth.plan.map(capital)
        opt=truth[truth.feasible].loc[truth[truth.feasible].objective.idxmin()]
        if np.isfinite(U[best]):assert L[best]-1e-8<=float(truth[truth.plan==sel].objective.iloc[0])<=U[best]+1e-8
        if np.isfinite(U[best]) and gap>0:assert sel==opt.plan
        result['certified_optimal']=bool(np.isfinite(U[best]) and gap>0)
        result.update(coverage_failures=0,classification_errors=0,reference_optimum=float(opt.objective))
        return result,pd.concat(audits),pd.DataFrame(self.trace)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--repeats',type=int,default=1);pa.add_argument('--coarse',type=int,default=3);pa.add_argument('--capital-scale',type=float,default=1.);pa.add_argument('--ablation',action='store_true');args=pa.parse_args()
    out=[]
    for repeat in range(args.repeats):
        for mode in (['direct_output'] if args.ablation else ['adaptive','uniform'] if repeat%2==0 else ['uniform','adaptive']):
            for family in FAMILIES:
                e=Experiment(family,mode,args.coarse,args.capital_scale);r,states,trace=e.run();r['repeat']=repeat;out.append(r)
                stem=f'{mode}{args.coarse}_{family}_c{args.capital_scale:g}_r{repeat}'
                states.to_csv(HERE/f'{stem}_states.csv',index=False);trace.to_csv(HERE/f'{stem}_trace.csv',index=False)
                print(json.dumps(r),flush=True)
    prefix='output_ablation' if args.ablation else f'adaptive{args.coarse}_nested_c{args.capital_scale:g}'
    (HERE/f'{prefix}_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame(out).to_csv(HERE/f'{prefix}_summary.csv',index=False)

if __name__=='__main__':main()
