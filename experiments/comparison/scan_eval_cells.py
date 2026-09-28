"""Run on each GPU host: print <dir>\t<eval_status>\t<mAP> for every eval dir under the given roots.

Usage: python3 scan_eval_cells.py ~/iraod_jobs ~/iraod_artifacts /mnt/shared/zechuan/iraod_artifacts > scan_<host>.tsv
"""
import os,sys,json,glob,re
roots=[r for r in sys.argv[1:] if os.path.isdir(r)]
seen=set()
walked=set()  # real dirs already walked; follows symlinks (ext_runs_e1 is a symlink on some hosts) without looping
for root in roots:
    for dp,dn,fn in os.walk(root, followlinks=True):
        dn[:]=[d for d in dn if d not in ('vis','tf_logs','.git','work','roi_features','images','JPEGImages','annfiles','Corruption')
               and os.path.realpath(os.path.join(dp,d)) not in walked]
        walked.update(os.path.realpath(os.path.join(dp,d)) for d in dn)
        if 'eval_status' in fn or 'metrics.json' in fn:
            rp=os.path.realpath(dp)
            if rp in seen: continue
            seen.add(rp)
            st=''
            try: st=open(os.path.join(dp,'eval_status')).read().strip().replace('\t',' ').replace('\n',' | ')
            except Exception: pass
            m=''
            js=sorted(glob.glob(os.path.join(dp,'eval_*.json')))+[os.path.join(dp,'metrics.json')]
            for j in js:
                try:
                    d=json.load(open(j)); mm=d.get('metric',d)
                    v=mm.get('mAP',mm.get('AP50',mm.get('mAP50')))
                    if v is not None: m=str(v)
                except Exception: pass
            md=os.path.dirname(dp)  # AASFOD standard recipe = per-seed TSD + two-stage run
            std='std2stage' if os.path.exists(os.path.join(md,'tsd.json')) and os.path.exists(os.path.join(md,'work','stages.json')) else ''
            print(f"{dp}\t{st}\t{m}\t{std}")
