"""Portable formal gates and exact independent hotspot coverage; synthetic only."""
from pathlib import Path
import importlib.util,copy
import pytest
import normalize as n
import render as r

def planner():
 path=Path(__file__).resolve().parents[1]/'campaign/state_final_plan/state_plan.py'
 spec=importlib.util.spec_from_file_location('public_hotspot_planner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def hotspot():
 return [dict(x,gpu_count=x['partitions'],grid_domain='partition-hotspot') for x in planner().hotspot_grid()]

def test_exact_hotspot32_admitted():n.validate_grid(hotspot(),'formal')
def test_missing_hotspot_cannot_be_full_grid():
 with pytest.raises(n.core.EvidenceError):n.validate_grid(hotspot()[:-1],'formal')
def test_hotspot_wrong_x_axis_rejected():
 rows=hotspot();rows[0]['hotspot_k']=8
 with pytest.raises(n.core.EvidenceError):n.validate_grid(rows,'formal')
def test_hotspot_cannot_masquerade_as_main_density_grid():
 rows=hotspot()
 for row in rows:row.pop('grid_domain')
 with pytest.raises(n.core.EvidenceError):n.validate_grid(rows,'formal')
def test_missing_formal_has_no_output(tmp_path):
 out=tmp_path/'paper'
 with pytest.raises(n.core.EvidenceError):r.render({'mode':'formal','paper_ready':False},out)
 assert not out.exists()
def test_qualification_has_no_paper_output(tmp_path):
 out=tmp_path/'paper'
 with pytest.raises(n.core.EvidenceError):r.render({'mode':'qualification','paper_ready':True},out)
 assert not out.exists()
def test_synthetic_cannot_publish(tmp_path):
 out=tmp_path/'paper'
 with pytest.raises(n.core.EvidenceError):r.publish({'synthetic_fixture':True,'paper_ready':True},out,tmp_path/'scratch')
 assert not out.exists()


def test_e4s_hash_only_cannot_authorize_formal_plot():
 with pytest.raises(n.core.EvidenceError,match='recoverable-output archive'):
  n.e4s_run(None,{},dict(accept_execution_hash_attestation=True),{}, {},'formal')
