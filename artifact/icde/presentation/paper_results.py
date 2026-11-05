"""Export paper E1 tables/figures only after all exact formal evidence passes."""
from pathlib import Path
import argparse,json
import e1_summary

def export(selection,output):
 selected=json.loads(Path(selection).read_text())
 if selected.get('mode')!='formal':raise ValueError('Paper export requires an explicit formal selection, never qualification')
 summary=e1_summary.summarize(Path(selection),'formal')
 if not summary['paper_ready']:raise ValueError('Incomplete formal cells/proofs: no paper files were generated')
 e1_summary.render(summary,Path(output));return summary

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.selection,a.output)
