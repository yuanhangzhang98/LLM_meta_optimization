#!/usr/bin/env python3
"""Recompute the saved campaign's final objective portfolio; no solver or API calls."""
import argparse,hashlib,json,sys
from pathlib import Path
SCRIPT=Path(__file__).resolve().parent
CORE=SCRIPT.parent
sys.path.insert(0,str(CORE))
sys.dont_write_bytecode=True  # Keep imported archived objectives read-only.

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,default=SCRIPT/'artifacts/data/campaign')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();campaign=a.campaign.resolve();output=a.output.resolve()
    if not (campaign/'database/database.json').is_file():ap.error('Extract supporting_data.zip into reproduce/artifacts/, or provide --campaign.')
    if output==campaign or campaign in output.parents:ap.error('Write output outside the archived campaign directory.')
    import workspace_config
    workspace_config.ROOT=campaign
    import consensus_objective as co
    from database.manager import DatabaseManager
    from meta_agent_state import get_effective_meta_weights
    # Code comes from this repository; the archived portfolio supplies inputs.
    for module in [workspace_config,co,sys.modules['database.manager']]:
        assert CORE in Path(module.__file__).resolve().parents,module.__file__
    db=DatabaseManager(campaign/'database/database.json');db.load()
    result={'designs':len(db.get_all_designs()),'consensus_source_sha256':hashlib.sha256((CORE/'consensus_objective.py').read_bytes()).hexdigest(),'variants':{}}
    for label,meta in [('raw',None),('final_recorded_meta',get_effective_meta_weights())]:
        co.invalidate_consensus_cache()
        fn,stats=co.create_consensus_objective(db,age_decay_base=.9,meta_weights=meta)
        scores={d.design_id:fn(d.experiments) for d in db.get_all_designs()}
        result['variants'][label]={'objectives':stats['num_objectives'],'weights':stats['weights'],'consensus_scores':scores,'top10_design_ids':sorted(scores,key=lambda d:(scores[d],d))[:10]}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Recomputed {result['designs']} designs × {result['variants']['raw']['objectives']} objectives; wrote {output}")
if __name__=='__main__':main()
