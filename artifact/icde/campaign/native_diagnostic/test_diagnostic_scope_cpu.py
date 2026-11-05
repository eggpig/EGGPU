import pytest
import native_worker as worker

def options(dataset,phase,count):
 return worker.parser().parse_args(['--algorithm','wcc','--dataset',dataset,'--phase',phase,'--count',str(count),'--gpu-uuids',','.join('GPU-'+str(i) for i in range(count))])

@pytest.mark.parametrize('dataset',['US','EP','LJ','OR','IN','UK','TW'])
def test_p1_diagnostic_all_graphs(dataset):
 assert worker.configure(options(dataset,'diagnostic',1)).dataset==dataset

@pytest.mark.parametrize('dataset',['EP','LJ'])
@pytest.mark.parametrize('phase',['target','formal'])
def test_no_silent_extension_of_main_e2(dataset,phase):
 with pytest.raises(ValueError,match='Frozen E2'):worker.configure(options(dataset,phase,1))

@pytest.mark.parametrize('count',[2,4,8])
def test_no_silent_extension_of_scaling(count):
 with pytest.raises(ValueError,match='Frozen E2'):worker.configure(options('EP','diagnostic',count))
