"""CPU-only recipe admission tests; generated examples are not GPU job evidence."""
import copy,json
from pathlib import Path
import pytest
import prepare_qualification as generator

def parent():
    recipe=json.loads(Path(__file__).with_name('qualification_recipe.json').read_text())
    rows=[]
    for row in recipe['jobs']:
        command=['SYNTHETIC_PYTHON','SYNTHETIC_WORKER','--execute-gpu']
        for key,value in row['algorithm_flags'].items():command.extend([key,value])
        rows.append(dict(row,id='wide_'+row['id'].removeprefix('wide3_'),
          depends_on=['wide_'+x.removeprefix('wide3_') for x in row['depends_on']],command=command,
          maintenance={'fixture_only':True}))
    return {'jobs':rows,'fixture_only':True}

def test_complete_80_recipe_preserves_original_four_scopes():
    plan=parent();assert generator.validate_parent_plan(plan)
    assert len(plan['jobs'])==80 and len([r for r in plan['jobs'] if r['phase']=='target'])==4
    assert all(len(r['depends_on'])==76 for r in plan['jobs'] if r['phase']=='target')

@pytest.mark.parametrize('mutation',['missing','duplicate','algorithm','timeout','maintenance','prerequisite','branch'])
def test_recipe_rejects_weaker_or_changed_parent(mutation):
    plan=copy.deepcopy(parent())
    if mutation=='missing':plan['jobs'].pop()
    elif mutation=='duplicate':plan['jobs'].append(plan['jobs'][0])
    elif mutation=='algorithm':plan['jobs'][4]['algorithm']='bfs'
    elif mutation=='timeout':plan['jobs'][-1]['timeout']=1
    elif mutation=='maintenance':del plan['jobs'][0]['maintenance']
    elif mutation=='prerequisite':plan['jobs'][-1]['depends_on'].pop()
    elif mutation=='branch':
        command=plan['jobs'][4]['command'];command[command.index('--wcc-qualification-variant')+1]='label_coo'
    with pytest.raises(AssertionError):generator.validate_parent_plan(plan)
