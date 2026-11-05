"""Real CPU C++/Python shared-clock roundtrip and malformed-event rejection."""
from pathlib import Path
import copy, json, os, subprocess, sys, tempfile, time, unittest
import sg_clock


class ClockContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = cls.root/'clock.cpp'
        source.write_text('#include "'+str(Path(__file__).with_name('sg_complete_clock.h'))+'"\n'
            '#include <vector>\n#include <thread>\n#include <chrono>\nint main(){'
            'for(int i=0;i<3;++i){std::vector<double> complete(16,1);'
            'eggpu_sg_complete_clock(i,complete.size());'
            'std::this_thread::sleep_for(std::chrono::milliseconds(20));}return 0;}\n')
        binary = cls.root/'clock'
        subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(binary)],check=True)
        cls.domain = sg_clock.clock_domain()
        cls.start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        # Required preparation belongs to the continuous interval, not a sum.
        time.sleep(.02)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', EGGPU_SG_START_NS=str(cls.start),
            EGGPU_SG_LAUNCH=str(cls.root/'launch.json'))
        with (cls.root/'native.log').open('w') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('sg_clock_exec.py')),str(binary)],
                stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        cls.end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        cls.launch = (cls.root/'launch.json').read_text()
        cls.log = (cls.root/'native.log').read_text()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        (self.root/'launch.json').write_text(self.launch)
        (self.root/'native.log').write_text(self.log)

    def result(self):
        return sg_clock.first_result(self.root,self.start,self.end,self.domain)

    def edit_row(self,index,key,value):
        rows=[json.loads(line.split(' ',1)[1]) for line in self.log.splitlines()]
        rows[index][key]=value
        (self.root/'native.log').write_text(''.join('SG_COMPLETE_MONOTONIC_JSON '+json.dumps(row)+'\n' for row in rows))

    def test_real_cpp_exec_shared_clock(self):
        row=self.result()
        self.assertEqual(row['values'],16)
        self.assertGreaterEqual(row['seconds'],.02)
        self.assertGreater((self.end-row['end_ns'])/1e9,.05)
        self.assertFalse(row['phase_durations_summed'])

    def test_wrong_native_pid(self):
        self.edit_row(0,'pid',0)
        with self.assertRaises(ValueError):self.result()

    def test_wrong_namespace(self):
        launch=json.loads(self.launch);launch['clock_domain']['time_namespace']='other'
        (self.root/'launch.json').write_text(json.dumps(launch))
        with self.assertRaises(ValueError):self.result()

    def test_duplicate_first_not_accepted(self):
        self.edit_row(1,'sequence',0)
        with self.assertRaises(ValueError):self.result()

    def test_missing_complete_event(self):
        (self.root/'native.log').write_text('\n'.join(self.log.splitlines()[1:]))
        with self.assertRaises(ValueError):self.result()

    def test_event_after_parent_end(self):
        self.edit_row(0,'t_ns',self.end+1)
        with self.assertRaises(ValueError):self.result()

    def test_changed_full_domain(self):
        self.edit_row(1,'values',15)
        with self.assertRaises(ValueError):self.result()

    def test_float_timestamp_rejected(self):
        self.edit_row(0,'t_ns',float(self.start+1))
        with self.assertRaises(ValueError):self.result()


if __name__=='__main__':unittest.main(verbosity=2)
