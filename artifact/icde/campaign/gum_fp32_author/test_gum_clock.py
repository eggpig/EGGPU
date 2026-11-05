"""CPU-only cross-process clock-domain and native marker evidence checks."""
from pathlib import Path
import copy,json,subprocess,sys,tempfile,time,unittest
import gum_clock as clock

class ClockContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.domain={'clock':'CLOCK_MONOTONIC','host':'synthetic'}
  for rank in range(2):
   launch=dict(rank=rank,size=2,pid=100+rank,clock_domain=self.domain,parent_monotonic_start_ns=1000,monotonic_launch_ns=1100)
   (self.root/f'launch-rank-{rank}.json').write_text(json.dumps(launch))
   rows=[]
   for r in range(3):
    begin=1200+r*1000+rank*10;end=begin+400+rank*100;identity=f'{100+rank}:{begin}'
    rows.extend([dict(kind='begin',phase='warmup_query' if r<2 else 'measured_query',query_id=r,rank=rank,pid=100+rank,id=identity,t_ns=begin),
                 dict(kind='end',pid=100+rank,id=identity,t_ns=end)])
   (self.root/f'memory-markers-rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def tearDown(self):self.temp.cleanup()
 def measure(self):return clock.first_result(self.root,2,1000,5000,self.domain)
 def test_continuous_endpoint_is_not_sum_of_phases_or_ranks(self):
  result=self.measure();self.assertEqual(result['start_ns'],1000);self.assertEqual(result['end_ns'],1710)
  self.assertEqual(result['seconds'],710/1e9);self.assertFalse(result['phase_durations_summed'])
  self.assertEqual(result['rank_zero_complete_ns'],1600)
 def test_different_time_namespace_rejected(self):
  p=self.root/'launch-rank-1.json';d=json.loads(p.read_text());d['clock_domain']['time_namespace']='different';p.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'CLOCK_MONOTONIC domain'):self.measure()
 def test_wrong_parent_origin_rejected(self):
  p=self.root/'launch-rank-0.json';d=json.loads(p.read_text());d['parent_monotonic_start_ns']=999;p.write_text(json.dumps(d))
  with self.assertRaises(ValueError):self.measure()
 def test_unmatched_end_rejected(self):
  p=self.root/'memory-markers-rank-0.jsonl';rows=p.read_text().splitlines();p.write_text('\n'.join(rows[1:])+'\n')
  with self.assertRaisesRegex(ValueError,'unique begin'):self.measure()
 def test_missing_rank_or_round_rejected(self):
  p=self.root/'memory-markers-rank-1.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:4])+'\n')
  with self.assertRaisesRegex(ValueError,'three complete'):self.measure()
 def test_marker_must_be_same_exec_pid(self):
  p=self.root/'memory-markers-rank-0.jsonl';p.write_text(p.read_text().replace('"pid": 100','"pid": 999'))
  with self.assertRaisesRegex(ValueError,'exec-bound rank'):self.measure()
 def test_actual_parent_child_clock_namespace_and_bracket(self):
  domain=clock.clock_domain();before=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
  code='import gum_clock,json,time; print(json.dumps({"domain":gum_clock.clock_domain(),"t":time.clock_gettime_ns(time.CLOCK_MONOTONIC)}))'
  child=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=Path(clock.__file__).parent,text=True))
  after=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
  self.assertEqual(domain,child['domain']);self.assertLessEqual(before,child['t']);self.assertLessEqual(child['t'],after)
 def test_parent_interval_must_be_valid(self):
  with self.assertRaises(ValueError):clock.first_result(self.root,2,5000,1000,self.domain)

if __name__=='__main__':unittest.main()
