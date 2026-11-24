"""Three independent complete repeats; never overwrite the original audit."""
from pathlib import Path
import json, statistics
import full_certificate_planning_study as study

HERE=Path(__file__).resolve().parent


def main():
    original_families=dict(study.FAMILIES)
    records=[]
    for repeat in range(1,4):
        study.OUT=HERE/f'full_certificate_planning_repeat{repeat}'
        if (study.OUT/'summary.json').exists():
            raise FileExistsError('Refusing to overwrite completed repeat: '+str(study.OUT))
        # Within the study, the second family reverses method order. Reversing
        # families therefore reverses both family-specific method orders too.
        study.FAMILIES=(dict(reversed(list(original_families.items())))
                        if repeat%2==0 else dict(original_families))
        study.main()
        result=json.loads((study.OUT/'summary.json').read_text())
        for total in result['totals']:
            record=dict(repeat=repeat,**total)
            record['certificate_including_offline_seconds']=(
                total['online_certificate_seconds']+total['offline_certificate_seconds'])
            records.append(record)
    summaries=[]
    quantities=['total_seconds','online_certificate_seconds',
                'certificate_including_offline_seconds','training_seconds',
                'offline_certificate_seconds','online_seconds',
                'menu_aggregation_seconds','common_selection_seconds']
    for method in ['full_current_corrected','compositional']:
        selected=[r for r in records if r['method']==method]
        summary=dict(method=method,repeats=len(selected))
        for key in quantities:
            values=[r[key] for r in selected]
            summary[key]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
        summaries.append(summary)
    ratios=[]
    for repeat in range(1,4):
        rows={r['method']:r for r in records if r['repeat']==repeat}
        full=rows['full_current_corrected'];comp=rows['compositional']
        ratios.append(dict(repeat=repeat,
             time_reduction=1-comp['total_seconds']/full['total_seconds'],
             speedup=full['total_seconds']/comp['total_seconds'],
             online_certificate_time_reduction=1-comp['online_certificate_seconds']/full['online_certificate_seconds'],
             offline_inclusive_certificate_time_reduction=1-comp['certificate_including_offline_seconds']/full['certificate_including_offline_seconds']))
    ratio_summary={key:dict(median=statistics.median(r[key] for r in ratios),
                           minimum=min(r[key] for r in ratios),maximum=max(r[key] for r in ratios))
                   for key in ratios[0] if key!='repeat'}
    result=dict(scope='Three additional independent complete runs; original raw run excluded from reported repeat medians. Every family and method independently pays all response training; composition additionally pays all tube certificates. All per-state correctness and same-prediction SHA checks are retained by the called runner.',
                source_runner='full_certificate_planning_study.py',
                method_order='Repeats 1 and 3: mixed9 full then composition, mixed30 composition then full. Repeat 2 reverses family order and each family method order.',
                records=records,method_summary=summaries,paired_ratios=ratios,paired_ratio_summary=ratio_summary)
    (HERE/'full_certificate_planning_repeats_summary.json').write_text(json.dumps(result,indent=2))
    print('REPEAT SUMMARY',json.dumps(dict(method_summary=summaries,paired_ratio_summary=ratio_summary)),flush=True)


if __name__=='__main__':main()
