import json
import math
from native_worker import json_metric_values


def test_documented_nonfinite_metric_sentinels_remain_explicit():
    raw={'unreachable_distance':float('inf'),'nested':[float('-inf'),float('nan')],
         'bytes':17,'seconds':0.25,'used':True}
    result=json_metric_values(raw)
    assert json.loads(json.dumps(result,allow_nan=False))==result
    assert result['unreachable_distance']=={'numeric_class':'positive_infinity'}
    assert result['nested']==[{'numeric_class':'negative_infinity'},{'numeric_class':'nan'}]
    assert result['bytes']==17 and result['seconds']==0.25 and result['used'] is True
    assert math.isinf(raw['unreachable_distance']) and math.isnan(raw['nested'][1])
