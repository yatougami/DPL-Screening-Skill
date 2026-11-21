"""Regression tests against the actual dpl.py, with zero real GPU jobs.

Only the three GPU predictor launch calls are simulated. Input converters,
score extractors and final screening run as actual source scripts. All receipts
for normal workflows are created by the tested controller itself.
"""
from __future__ import annotations
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ['DPL_TEST_OUTPUT']).resolve() if os.environ.get('DPL_TEST_OUTPUT') else Path(tempfile.mkdtemp(prefix='dpl-v4-tests-'))
OUT.mkdir(parents=True,exist_ok=True)
SCRIPTS=ROOT/'.agents/skills/dpl-screening/scripts'
TARGET=SCRIPTS/'dpl.py'
CASE_ROOT=OUT/'cases'
CASE_ROOT.mkdir(exist_ok=True)
(OUT/'tmp').mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'))
sys.dont_write_bytecode=True
import numpy as np
import yaml
from rdkit import Chem


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


dpl=load('tested_dpl',TARGET)
REAL_RUN=dpl.run_cmd
SOURCE_HASH=hashlib.sha256(TARGET.read_bytes()).hexdigest()
RESULTS=[]
CURRENT=None
EVENTS=[]
SIM=None
IDS=['P00','P01','P02','P03','P04']
VALUES={key:value for key,value in zip(IDS,[(.9,.9,.1),(.8,.8,.9),(.3,.3,.8),(.2,.2,.2),(.1,.1,.1)])}
PREDICTORS={'run_alphafold.py':'af3','run_boltz2_batch.py':'boltz2','run_chai_batch.py':'chai1'}


def js(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()} if root.exists() else {}


def at(work, relative):
    """Map test paths to the workdir layout. R27 checks the layout independently."""
    parts=Path(relative).parts
    category=parts[0]
    rest=Path(*parts[1:])
    if category=='inputs':return dpl.model_input(work,parts[1])/Path(*parts[2:])
    if category=='predictions':return dpl.model_output(work,parts[1])/Path(*parts[2:])
    if category=='scores':
        if parts[1]=='scores_manifest.json':return dpl.info(work)/parts[1]
        model=parts[1].removesuffix('_scores.csv')
        return dpl.score_path(work,model)
    if category=='results':return dpl.reports(work)/rest
    return dpl.info(work)/relative


def input_hashes(work):
    return {m:hashes(dpl.model_input(work,m)) for m in dpl.MODELS}


def score_hashes(work):
    paths=[dpl.score_path(work,m) for m in dpl.MODELS]+[dpl.info(work)/'scores_manifest.json']
    return {str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


from synthetic_outputs import predictions as make_predictions


class PredictorSimulator:
    def __init__(self,folder,*,failure_model=None,failure_after=0,omit_af3_conf=False,bad_boltz_npz=False):
        self.folder=folder
        self.launches=[]
        self.failure_model=failure_model
        self.failure_after=failure_after
        self.omit_af3_conf=omit_af3_conf
        self.bad_boltz_npz=bad_boltz_npz

    def run(self,model,cmd,env):
        def option(*names):
            key=next(n for n in names if n in cmd)
            return Path(cmd[cmd.index(key)+1])
        inputs=option('--input_dir','-i')
        outputs=option('--output_dir','-o')
        suffix=dpl.EXTS[model]
        tasks=sorted(p.stem for p in inputs.glob('*'+suffix))
        self.launches.append(dict(model=model,tasks=tasks,argv=cmd,cuda=(env or {}).get('CUDA_VISIBLE_DEVICES')))
        failures = {}
        for index,key in enumerate(tasks):
            if model==self.failure_model and index==self.failure_after:
                partial=outputs/'_tmp_boltz2'/key/'predict_output/partial.txt'
                partial.parent.mkdir(parents=True,exist_ok=True)
                partial.write_text('Synthetic partial output; no GPU process ran.',encoding='utf-8')
                failures[key] = 'simulated sample failure'
                continue
            with tempfile.TemporaryDirectory(prefix='s-',dir=OUT/'tmp') as temp:
                scratch=Path(temp)
                fixture_key='synthetic' if model=='chai1' else key
                make_predictions(scratch,fixture_key,VALUES.get(key,(.8,.8,.8)))
                source=scratch/'predictions'/model/fixture_key
                dest=outputs/key
                if model=='boltz2':
                    dest=outputs/('boltz_results_'+inputs.stem)/'predictions'/key
                    source=source/'predictions'/key
                shutil.copytree(source,dest)
                if model=='af3':
                    shutil.copy2(inputs/(key+suffix),dest/(key+'_data.json'))
                    if self.omit_af3_conf:
                        for p in dest.rglob('*confidences.json'):
                            if 'summary' not in p.name:p.unlink()
                elif model=='boltz2':
                    if self.bad_boltz_npz:
                        np.savez(dest/f'plddt_{key}_model_0.npz',plddt=np.array([.8]))

        js(Path(env['DPL_BATCH_ERRORS']), failures)
        if failures:
            raise subprocess.CalledProcessError(1,cmd)


def dispatch(cmd,log=None,env=None,dry=False):
    child=Path(cmd[1]).name if len(cmd)>1 else ''
    model=PREDICTORS.get(child)
    EVENTS.append(dict(argv=list(map(str,cmd)),dry=dry,model=model))
    if model:
        if dry:return REAL_RUN(cmd,log,env,True)
        if SIM is None:raise RuntimeError('AUDIT GUARD: real model launch is prohibited')
        return SIM.run(model,list(cmd),env)
    allowed={SCRIPTS/'screen.py'}|set((SCRIPTS/'legacy').glob('*/csv_to_*.py'))|set((SCRIPTS/'legacy').glob('*/score_to_csv.py'))
    if len(cmd)<2 or Path(cmd[1]) not in allowed:
        raise RuntimeError('AUDIT GUARD: unexpected subprocess: '+repr(cmd))
    return REAL_RUN(cmd,log,env,dry)


def invoke(*arguments,sim=None):
    global SIM
    previous=SIM;SIM=sim
    try:
        with patch.object(sys,'argv',[str(TARGET),*map(str,arguments)]),patch.object(dpl,'run_cmd',side_effect=dispatch):
            return dpl.main()
    finally:SIM=previous


def trial(case_id,related,title,fn):
    global CURRENT,EVENTS
    folder=CASE_ROOT/case_id
    if folder.resolve().parent!=CASE_ROOT.resolve():raise RuntimeError('Unsafe audit output path')
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir()
    CURRENT=folder;EVENTS=[]
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):detail=fn(folder)
        result=dict(id=case_id,related=related,title=title,executed=True,detail=detail)
    except Exception:
        result=dict(id=case_id,related=related,title=title,executed=False,harness_error=traceback.format_exc())
    (folder/'execution.txt').write_text(captured.getvalue(),encoding='utf-8')
    js(folder/'commands.json',EVENTS)
    js(folder/'observation.json',result)
    RESULTS.append(result)
    print(case_id,'EXECUTED' if result['executed'] else 'HARNESS_ERROR',flush=True)


def normalize(folder,ids=None,sequences=None,extra_original=None,smiles='CCO'):
    folder.mkdir(parents=True,exist_ok=True)
    ids=list(IDS if ids is None else ids)
    sequences=sequences or ['AC']*len(ids)
    rows=[dict(task_id=key,Protein_seq=seq,ligand_smiles=smiles) for key,seq in zip(ids,sequences)]
    cols=list(dpl.REQUIRED)
    if extra_original is not None:
        cols+=['original_protein_id']
        for row,original in zip(rows,extra_original):row['original_protein_id']=original
    source=folder/'source.csv';dpl.write_csv(source,rows,cols)
    work=folder/'run'
    _,config_path=configuration(folder)
    assert invoke('normalize','--input',source,'--workdir',work,'--config',config_path)==0
    return work,source


def configuration(folder):
    cfg=read(ROOT/'config/server.example.json')
    runtime=folder/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    for m in dpl.MODELS:cfg[m]['python']=sys.executable
    for m,key in [('af3','model_dir'),('boltz2','cache_dir'),('chai1','weights_dir')]:
        p=runtime/(m+'_'+key);p.mkdir(exist_ok=True);cfg[m][key]=str(p)
    for key in ('checkpoint_path','affinity_checkpoint_path'):
        p=runtime/(key+'.ckpt');p.write_bytes(b'Audit path placeholder; no real model weights.')
        cfg['boltz2'][key]=str(p)
    path=folder/'config.json';js(path,cfg)
    return cfg,path


def prepared(folder,ids=None,sequences=None,extra_original=None):
    work,source=normalize(folder,ids,sequences,extra_original)
    assert invoke('prepare','--workdir',work)==0
    cfg,path=configuration(folder)
    return work,source,cfg,path


def pipeline(folder,ids=None):
    work,source,cfg,path=prepared(folder,ids)
    sim=PredictorSimulator(folder)
    assert invoke('predict','--workdir',work,'--config',path,sim=sim)==0
    assert invoke('collect','--workdir',work,'--config',path)==0
    assert invoke('screen','--workdir',work)==0
    return work,source,cfg,path,sim


def cloned(folder):
    base=CASE_ROOT/'R01_normal_pipeline'
    shutil.copytree(base/'run',folder/'run')
    shutil.copy2(base/'config.json',folder/'config.json')
    return folder/'run',read(folder/'config.json'),folder/'config.json'


def external_scores(folder,n=10,arrays=None,seq='AC'):
    ids=[f'P{i:02d}' for i in range(n)]
    work,_=normalize(folder,ids,[seq]*n)
    arrays=arrays or [list(reversed(range(1,n+1)))]*3
    for m,values in zip(dpl.MODELS,arrays):
        rule=dpl.SCORE_RULES[m]
        rows=[]
        for key,value in zip(ids,values):
            row=dict(task_id=key,Protein_seq=seq,ligand_smiles='CCO')
            row.update({rule['mode']:'single',rule['status']:'OK',rule['score']:value,rule['sample']:'0'})
            if 'seed' in rule:row[rule['seed']]='1'
            if m=='boltz2':row['boltz2_sample_status']='model_0:OK'
            rows.append(row)
        dpl.write_csv(dpl.score_path(work,m),rows,list(rows[0]))
    return work,ids


def r01(folder):
    work,source,cfg,path,sim=pipeline(folder)
    selected,_=dpl.load_csv(at(work,'results/selected_sequences.csv'))
    actual=[r['task_id'] for r in selected]
    assert actual==['P00','P01','P02'],actual
    for m in dpl.MODELS:
        assert all(dpl.trusted_complete(m,work,cfg,key) for key in IDS)
    summary=read(at(work,'results/screening_summary.json'))
    return dict(passed=True,selected=actual,excluded_count=summary['excluded_count'],
                predictor_processes_simulated=len(sim.launches),real_gpu_processes=0,
                target_sha256=SOURCE_HASH,actual_extractors_executed=True,
                controller_generated_receipts=True,expected_reduction_fraction=.4)


def r02(folder):
    work,source,cfg,path=prepared(folder)
    first=input_hashes(work)
    a=invoke('prepare','--workdir',work)
    b=invoke('all','--input',source,'--workdir',work,'--config',path,'--dry-run')
    c=invoke('all','--input',source,'--workdir',work,'--config',path,'--dry-run')
    sim=PredictorSimulator(folder)
    d=invoke('all','--input',source,'--workdir',work,'--config',path,sim=sim)
    second_sim=PredictorSimulator(folder)
    e=invoke('predict','--workdir',work,'--config',path,sim=second_sim)
    return dict(passed=all(x==0 for x in (a,b,c,d,e)) and first==input_hashes(work) and not second_sim.launches,
        exit_codes=[a,b,c,d,e],input_files_unchanged=first==input_hashes(work),
        repeated_predict_launches=second_sim.launches)


def r03(folder):
    old=CASE_ROOT/'R01_normal_pipeline/run'
    observations=[]
    for variant in ('different_sequence','different_ligand','different_model_config','missing_receipts'):
        sub=folder/variant
        sequences=['GGGG']*5 if variant=='different_sequence' else ['AC']*5
        work,source=normalize(sub,sequences=sequences,smiles='CCC' if variant=='different_ligand' else 'CCO')
        assert invoke('prepare','--workdir',work)==0
        cfg,path=configuration(sub)
        if variant=='different_model_config':cfg['af3']['model_dir']+='_different';Path(cfg['af3']['model_dir']).mkdir();js(path,cfg)
        for model in dpl.MODELS:
            shutil.copytree(dpl.model_output(old,model),dpl.model_output(work,model),dirs_exist_ok=True)
        if variant!='missing_receipts':shutil.copytree(at(old,'provenance'),at(work,'provenance'),dirs_exist_ok=True)
        trusted={m:all(dpl.trusted_complete(m,work,cfg,key) for key in IDS) for m in dpl.MODELS}
        code=invoke('collect','--workdir',work,'--config',path)
        observations.append(dict(variant=variant,collect_exit=code,trusted=trusted,
                                 passed=code!=0 and (not any(trusted.values()) if variant!='different_model_config' else not trusted['af3'])))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r04(folder):
    work,source,cfg,path=prepared(folder)
    p=at(work,'inputs/af3/P00.json');obj=read(p);obj['modelSeeds']=[99];obj['sequences'][0]['protein']['sequence']='GGGG';js(p,obj)
    sim=PredictorSimulator(folder)
    a=invoke('predict','--workdir',work,'--config',path,'--dry-run',sim=sim)
    b=invoke('prepare','--workdir',work)
    return dict(passed=a!=0 and b!=0 and not sim.launches,predict_exit=a,prepare_exit=b,
                simulator_launches=len(sim.launches),tampered_file_preserved=p.exists())


def r05(folder):
    checks=[]
    for variant in ('changed_output','changed_receipt','deleted_receipt','wrong_config'):
        sub=folder/variant;sub.mkdir()
        work,cfg,path=cloned(sub)
        if variant=='changed_output':
            p=at(work,'predictions/chai1/P00/seed_1/pred.model_idx_0.cif')
            p.write_text(p.read_text()+'\n# changed output\n',encoding='utf-8')
        elif variant=='changed_receipt':
            p=dpl.receipt_path(work,'chai1','P00');obj=read(p);obj['recipe']['protein_sequence']='GGGG';js(p,obj)
        elif variant=='deleted_receipt':dpl.receipt_path(work,'af3','P00').unlink()
        else:
            cfg['chai1']['weights_dir']+='_changed';Path(cfg['chai1']['weights_dir']).mkdir();js(path,cfg)
        code=invoke('collect','--workdir',work,'--config',path)
        checks.append(dict(variant=variant,exit_code=code,passed=code!=0))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)


def r06(folder):
    root=folder/'outputs';task='P01'
    p=root/task/'seed_1';p.mkdir(parents=True)
    (p/'pred.model_idx_0.cif').touch();(p/'scores.model_idx_0.npz').touch()
    chai=dpl.complete('chai1',root,task)
    for i in range(5):
        p=root/task/f'seed-1_sample-{i}';p.mkdir(parents=True)
        (p/'empty_model.cif').touch();(p/'empty_summary_confidences.json').touch()
    af3=dpl.complete('af3',root,task)
    p=root/task/'predictions'/task;p.mkdir(parents=True)
    for name in (f'{task}_model_0.pdb',f'confidence_{task}_model_0.json',f'plddt_{task}_model_0.npz',
                 f'pae_{task}_model_0.npz',f'pde_{task}_model_0.npz',f'affinity_{task}.json'):
        (p/name).write_bytes(b'invalid model data')
    boltz=dpl.complete('boltz2',root,task)
    fake=folder/'zip_with_invalid_npy.npz'
    with zipfile.ZipFile(fake,'w') as z:z.writestr('array.npy',b'not a NumPy array')
    accepted=dpl.valid_npz(fake)
    return dict(original_reproduction_fixed=not any((chai,af3,boltz)),
                empty_chai_accepted=chai,empty_af3_accepted=af3,invalid_boltz_accepted=boltz,
                residual_invalid_npy_payload_accepted=accepted,
                passed=not any((chai,af3,boltz,accepted)))


def r07(folder):
    observations=[]
    for variant in ('missing_af3_confidences','invalid_boltz_dimensions'):
        sub=folder/variant
        work,source,cfg,path=prepared(sub,ids=['P01'])
        sim=PredictorSimulator(sub,omit_af3_conf=variant=='missing_af3_confidences',bad_boltz_npz=variant=='invalid_boltz_dimensions')
        prediction=invoke('predict','--workdir',work,'--config',path,sim=sim)
        collection=invoke('collect','--workdir',work,'--config',path)
        repeat=PredictorSimulator(sub)
        retry=invoke('predict','--workdir',work,'--config',path,sim=repeat)
        observations.append(dict(variant=variant,predict_exit=prediction,collect_exit=collection,retry_exit=retry,
             retry_launches=repeat.launches,repair_targeted=prediction==0 and collection!=0 and retry==0 and len(repeat.launches)==1))
    return dict(passed=all(x['predict_exit'] == 0 and x['collect_exit'] != 0 and x['repair_targeted'] for x in observations),observations=observations)


def r08(folder):
    observations=[]
    for variant in ('missing_score_file','input_changed','all_missing_input'):
        sub=folder/variant;sub.mkdir()
        work,cfg,path=cloned(sub)
        old=hashes(at(work,'results'))
        if variant=='missing_score_file':
            (at(work,'scores/af3_scores.csv')).unlink()
            code=invoke('screen','--workdir',work)
        elif variant=='input_changed':
            p=at(work,'input.csv');p.write_text(p.read_text(encoding='utf-8-sig').replace('AC,','GG,'),encoding='utf-8-sig')
            code=invoke('screen','--workdir',work)
        else:code=invoke('all','--input',sub/'missing.csv','--workdir',work,'--config',path)
        backups=list((at(work,'backups/results')).glob('reports_*'))
        preserved=any(hashes(p)==old for p in backups)
        observations.append(dict(variant=variant,exit_code=code,current_results_exist=(at(work,'results')).exists(),
            archived_results_preserved=preserved,workflow_status=read(at(work,'workflow_status.json')),
            passed=code!=0 and not (at(work,'results')).exists() and preserved))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r09(folder):
    ids=['0001','0002','NA','NULL','1e3','a.b','a__c']
    work,source,cfg,path,sim=pipeline(folder,ids)
    normalized,_=dpl.load_csv(at(work,'input.csv'))
    selected,_=dpl.load_csv(at(work,'results/selected_sequences.csv'))
    actual_originals=[r['original_protein_id'] for r in selected]
    for m in dpl.MODELS:
        scores,_=dpl.load_csv(dpl.score_path(work,m))
        assert {r['task_id'] for r in scores}=={r['task_id'] for r in normalized}
    return dict(passed=set(actual_originals)==set(ids),original_ids=ids,
        normalized_ids=[r['task_id'] for r in normalized],selected_original_ids=actual_originals,
        treatment='Numeric/NA IDs receive a safe protein_ alias; the legacy Chai extractor itself is unchanged.')


def r10(folder):
    observations=[]
    sources={'txt_unnamed':('input.txt','>\nACDEFG\n>good\nACDEFGHIK\n',0,2),
             'fasta_unnamed':('input.fasta','>\nACDEFG\n>good\nACDEFGHIK\n',0,2),
             'empty_record':('input.txt','>empty\n>good\nACDEFGHIK\n',1,None)}
    for variant,(filename,text,expected,count) in sources.items():
        sub=folder/variant;sub.mkdir();p=sub/filename;p.write_text(text,encoding='utf-8')
        work=sub/'run';_,cfgpath=configuration(sub);code=invoke('normalize','--input',p,'--smiles','CCO','--workdir',work,'--config',cfgpath)
        rows=dpl.load_csv(at(work,'input.csv'))[0] if (at(work,'input.csv')).exists() else []
        observations.append(dict(variant=variant,exit_code=code,rows=rows,
          passed=code==expected and (len(rows)==count if count is not None else not rows)))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r11(folder):
    observations=[]
    mutations=['mean5','wrong_seed','wrong_sample','partial_detailed','invalid_status','missing_score','nan','inf','duplicate','missing_metadata']
    for variant in mutations:
        sub=folder/variant
        work,ids=external_scores(sub,3)
        model='boltz2' if variant=='partial_detailed' else 'af3'
        p=dpl.score_path(work,model);rows,cols=dpl.load_csv(p);rule=dpl.SCORE_RULES[model]
        if variant=='mean5':rows[0][rule['mode']]='mean5';rows[0][rule['status']]='OK_MEAN5'
        elif variant=='wrong_seed':rows[0][rule['seed']]='99'
        elif variant=='wrong_sample':rows[0][rule['sample']]='4'
        elif variant=='partial_detailed':rows[0]['boltz2_sample_status']='model_0:PARTIAL_MISSING_pre_affinity'
        elif variant=='invalid_status':rows[0][rule['status']]='FAILED'
        elif variant=='missing_score':rows[0][rule['score']]=''
        elif variant in ('nan','inf'):rows[0][rule['score']]=variant
        elif variant=='duplicate':rows.append(rows[0].copy())
        elif variant=='missing_metadata':cols.remove(rule['mode'])
        dpl.write_csv(p,rows,cols)
        code=invoke('screen','--workdir',work,'--allow-unverified-scores')
        observations.append(dict(variant=variant,exit_code=code,passed=code!=0 and not (at(work,'results')).exists()))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r12(folder):
    work,ids=external_scores(folder,3)
    default=invoke('screen','--workdir',work)
    explicit=invoke('screen','--workdir',work,'--allow-unverified-scores')
    return dict(passed=default!=0 and explicit==0,default_exit=default,explicit_external_exit=explicit,
                documented_exception=True,external_tables_still_pass_sample_and_status_QC=True)


def r13(folder):
    observations=[]
    for variant in ('missing_boltz_python','missing_chai_weights','missing_af3_weights','checkpoint_is_directory'):
        sub=folder/variant
        work,source,cfg,path=prepared(sub)
        if variant=='missing_boltz_python':cfg['boltz2']['python']=str(sub/'missing_python')
        elif variant=='missing_chai_weights':cfg['chai1']['weights_dir']=str(sub/'missing_weights')
        elif variant=='missing_af3_weights':cfg['af3']['model_dir']=str(sub/'missing_weights')
        else:
            p=sub/'checkpoint_directory';p.mkdir();cfg['boltz2']['checkpoint_path']=str(p)
        js(path,cfg)
        sim=PredictorSimulator(sub)
        code=invoke('predict','--workdir',work,'--config',path,sim=sim)
        observations.append(dict(variant=variant,exit_code=code,launches=len(sim.launches),
            passed=code!=0 and not sim.launches))
    return dict(passed=all(x['passed'] for x in observations),
        original_missing_paths_fixed=all(x['passed'] for x in observations[:3]),observations=observations)


def r14(folder):
    work,source,cfg,path=prepared(folder)
    failed=PredictorSimulator(folder,failure_model='boltz2',failure_after=1)
    first=invoke('predict','--workdir',work,'--config',path,sim=failed)
    audit=invoke('collect','--workdir',work,'--config',path)
    partial=list((dpl.info(work)/'batches').rglob('partial.txt'))
    successes={key:dpl.trusted_complete('boltz2',work,cfg,key) for key in IDS}
    recovery=PredictorSimulator(folder)
    second=invoke('predict','--workdir',work,'--config',path,sim=recovery)
    third=invoke('collect','--workdir',work,'--config',path)
    fourth=invoke('screen','--workdir',work)
    boltz_launch=next(x for x in recovery.launches if x['model']=='boltz2')
    skip_completed='P00' not in boltz_launch['tasks']
    keep_tmp=all('--keep-tmp' in x['argv'] for x in failed.launches+recovery.launches if x['model']=='boltz2')
    return dict(passed=first!=0 and audit!=0 and all(x==0 for x in (second,third,fourth)) and bool(partial) and skip_completed and keep_tmp,
        exit_codes=[first,second,third,fourth],partial_files=[str(p) for p in partial],
        trusted_jobs_after_failure=successes,recovery_launches=recovery.launches,
        keep_tmp_flag=keep_tmp,already_successful_tasks_not_rerun=skip_completed)


def r15(folder):
    runner=load('tested_original_boltz_runner',SCRIPTS/'legacy/boltz2/run_boltz2_batch.py')
    work,source,cfg,path=prepared(folder,ids=IDS[:3])
    evidence=[]
    def fake_native(command,env=None):
        assert command[1:3]==['predict',str(at(work,'inputs/boltz2'))]
        assert len(list(Path(command[2]).glob('*.yaml')))==3
        assert all(command[command.index(flag)+1]=='1' for flag in ('--devices','--seed','--diffusion_samples','--diffusion_samples_affinity'))
        p=Path(command[command.index('--out_dir')+1])/'partial_prediction.txt'
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('CPU simulation failure evidence',encoding='utf-8')
        evidence.append(p)
        return subprocess.CompletedProcess(command,1)
    argv=['runner','-i',str(at(work,'inputs/boltz2')),'-o',str(folder/'runner_output'),
          '--checkpoint-path',cfg['boltz2']['checkpoint_path'],'--affinity-checkpoint-path',cfg['boltz2']['affinity_checkpoint_path'],
          '--cache-dir',cfg['boltz2']['cache_dir'],'--diffusion-samples','1','--diffusion-samples-affinity','1','--keep-tmp']
    with (patch.object(sys,'argv',argv),patch.object(runner,'require_boltz2',return_value=('2.0.0',folder/'unused_boltz')),
          patch.object(runner.subprocess,'run',side_effect=fake_native) as native):
        code=runner.main()
    return dict(passed=code==1 and native.call_count==1 and evidence[0].is_file(),
                native_processes=native.call_count,partial_output_preserved=evidence[0].is_file(),argv=argv)


def r16(folder):
    work,source=normalize(folder,ids=['a.b','a__c','A_B'],extra_original=['a.b','a__c','A/B'])
    rows,_=dpl.load_csv(at(work,'input.csv'))
    return dict(passed=[r['task_id'] for r in rows]==['a.b','a__c','A_B'] and rows[2]['original_protein_id']=='A/B',
                dot_preserved=rows[0]['task_id']=='a.b',repeated_underscores_preserved=rows[1]['task_id']=='a__c',
                provided_original_id_preserved=rows[2]['original_protein_id']=='A/B',rows=rows)


def r17(folder):
    arrays=[[10,9,8,7,6,5,4,3,2,1],[1,2,10,9,8,7,6,5,4,3],[1,2,3,4,10,9,8,7,6,5]]
    work,ids=external_scores(folder/'manual_union',10,arrays,seq='ACDEFGHIKLMNPQRSTVWY'*11)
    a=invoke('screen','--workdir',work,'--allow-unverified-scores')
    selected,_=dpl.load_csv(at(work,'results/selected_sequences.csv'))
    records,_=dpl.table_from_file(at(work,'results/selected_sequences.fasta'))
    checks={'manual_union':a==0 and {r['task_id'] for r in selected}==set(ids[:7]),
            'full_220aa_fasta':len(records)==7 and all(r['Protein_seq']=='ACDEFGHIKLMNPQRSTVWY'*11 for r in records)}
    for n in (1,2,3,4,10):
        w,keys=external_scores(folder/f'all_ties_{n}',n,[[.5]*n]*3)
        code=invoke('screen','--workdir',w,'--allow-unverified-scores')
        summary=read(at(w,'results/screening_summary.json'))
        checks[f'all_ties_{n}']=code==0 and summary['selected_count']==n
    w,keys=external_scores(folder/'boundary',6,[[.9,.8,.8,.1,.1,.1],[.1,.1,.1,.9,.8,.1],[.1,.1,.1,.1,.9,.8]])
    code=invoke('screen','--workdir',w,'--allow-unverified-scores')
    summary=read(at(w,'results/screening_summary.json'))
    checks['boundary_all_ties']=code==0 and summary['selection']['af3']['actual_retained']==3 and summary['selected_count']==6
    return dict(passed=all(checks.values()),checks=checks,external_source_mode_explicit_for_math_fixtures=True)


def r18(folder):
    observations=[]
    for variant in ('missing_map','null_map','empty_map'):
        sub=folder/variant;sub.mkdir();work,cfg,path=cloned(sub)
        p=at(work,'scores/scores_manifest.json');data=read(p)
        if variant=='missing_map':data.pop('receipt_sha256')
        elif variant=='null_map':data['receipt_sha256']=None
        else:data['receipt_sha256']={}
        js(p,data)
        # No explicit external/unverified-score permission is supplied.
        for model in dpl.MODELS:shutil.rmtree(dpl.model_output(work,model))
        shutil.rmtree(at(work,'provenance'))
        code=invoke('screen','--workdir',work)
        observations.append(dict(variant=variant,exit_code=code,new_results_exist=(at(work,'results')).exists(),
             default_mode=True,predictions_absent=True,receipts_absent=True,
             passed=code!=0 and not (at(work,'results')).exists()))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r19(folder):
    work,cfg,path=cloned(folder)
    before=score_hashes(work)
    code=invoke('collect','--workdir',work,'--config',path,'--dry-run')
    after=score_hashes(work)
    screening=invoke('screen','--workdir',work)
    return dict(passed=code==0 and before==after and screening==0,dry_run_exit=code,
                scoring_files_unchanged=before==after,deleted_files=sorted(set(before)-set(after)),
                subsequent_screen_exit=screening)


def r20(folder):
    work,cfg,path=cloned(folder)
    before=hashes(at(work,'results'))
    previous=read(at(work,'workflow_status.json'))
    def failing_dispatch(cmd,log=None,env=None,dry=False):
        if Path(cmd[1])==SCRIPTS/'legacy/boltz2/score_to_csv.py':
            EVENTS.append(dict(argv=list(cmd),injected_cpu_extractor_failure=True))
            raise subprocess.CalledProcessError(1,cmd)
        return dispatch(cmd,log,env,dry)
    with patch.object(dpl,'run_cmd',side_effect=failing_dispatch),patch.object(sys,'argv',[str(TARGET),'collect','--workdir',str(work),'--config',str(path)]):
        code=dpl.main()
    current=read(at(work,'workflow_status.json'))
    stale=before==hashes(at(work,'results')) and current==previous and current.get('status')=='success'
    return dict(passed=code!=0 and not stale,exit_code=code,old_results_unchanged=before==hashes(at(work,'results')),
                workflow_status=current,score_manifest_exists=(at(work,'scores/scores_manifest.json')).exists(),
                stale_success_reproduced=stale)


def r21(folder):
    skill=ROOT/'.agents/skills/dpl-screening'
    targets={name:(ROOT/name).read_text(encoding='utf-8') for name in ('README.md','tests/test_workflow.py')}
    targets['SKILL.md']=(skill/'SKILL.md').read_text(encoding='utf-8')
    entry={name:dict(references_current_controller='dpl.py' in text)
           for name,text in targets.items()}
    layout=dict(dependencies_in_skill=(skill/'assets/requirements_prepare.txt').is_file(),
                config_at_project=(ROOT/'config/server.example.json').is_file())
    return dict(passed=all(x['references_current_controller'] for x in entry.values()) and layout['config_at_project'] and layout['dependencies_in_skill'],
                entry=entry,layout=layout,no_installation_performed=True)


def r22(folder):
    p=folder/'source.fasta';p.write_text('>P01\nAC\n',encoding='utf-8')
    _,cfgpath=configuration(folder)
    absent={'usable':True,'version':[3,12,0],'requested_python':sys.executable,
            'modules':{name:{'available':name!='rdkit.Chem'} for name in dpl.MODULES}}
    with patch.object(dpl,'probe_python',return_value=absent):
        result=invoke('normalize','--input',p,'--smiles','CCO','--config',cfgpath,'--workdir',folder/'without_rdkit','--env-policy','reuse')
    requirements=(ROOT/'requirements_prepare.txt').read_text(encoding='utf-8')
    mandatory=[line.strip().lower() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    declared=any(line.startswith('rdkit') for line in mandatory)
    return dict(passed=result!=0 and declared and not (folder/'without_rdkit/run_info/input.csv').exists(),
                missing_rdkit_exit=result,mandatory_dependency_declared=declared,
                real_rdkit_validates_input=Chem.MolFromSmiles('CCO') is not None)


def r23(folder):
    observations=[]
    samples={
      'bad_sequence':'task_id,Protein_seq,ligand_smiles\nP01,ACDX,CCO\n',
      'multiple_ligands':'task_id,Protein_seq,ligand_smiles\nP01,ACD,CCO\nP02,ACD,CCC\n',
      'invalid_smiles':'task_id,Protein_seq,ligand_smiles\nP01,ACD,NOT_A_SMILES\n',
      'alias_collision':'task_id,Protein_seq,ligand_smiles\n0001,ACD,CCO\nprotein_0001,ACD,CCO\n',
      'ambiguous_headers':'task_id,ID,Protein_seq,ligand_smiles\nP01,Q01,ACD,CCO\n',
      'duplicate_headers':'task_id,Protein_seq,Protein_seq,ligand_smiles\nP01,ACD,ACD,CCO\n',
    }
    for name,text in samples.items():
        sub=folder/name;sub.mkdir();p=sub/'source.csv';p.write_text(text,encoding='utf-8')
        _,cfgpath=configuration(sub)
        code=invoke('normalize','--input',p,'--workdir',sub/'run','--config',cfgpath)
        observations.append(dict(variant=name,exit_code=code,passed=code!=0))
    return dict(passed=all(x['passed'] for x in observations),observations=observations)


def r24(folder):
    work,source,cfg,path=prepared(folder)
    cfg['execution']['mode']='parallel'
    for i,m in enumerate(dpl.MODELS):cfg[m]['gpu']=i
    js(path,cfg)
    sim=PredictorSimulator(folder)
    rejected=invoke('predict','--workdir',work,'--config',path,sim=sim)
    assert rejected and not sim.launches
    cfg['execution']['mode']='sequential';js(path,cfg)
    assert invoke('predict','--workdir',work,'--config',path,sim=sim)==0
    assert invoke('collect','--workdir',work,'--config',path)==0
    for i,m in enumerate(dpl.MODELS):cfg[m]['gpu']=i+3
    js(path,cfg)
    resume=PredictorSimulator(folder)
    code=invoke('predict','--workdir',work,'--config',path,sim=resume)
    return dict(passed=code==0 and not resume.launches and [x['model'] for x in sim.launches]==list(dpl.MODELS),
                parallel_rejected=True,hardware_only_change_reused=True)


def r25(folder):
    files=[]
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));files.append(p.relative_to(ROOT).as_posix())
    result=subprocess.run([sys.executable,'-B',str(TARGET),'--help'],cwd=OUT,env=os.environ.copy(),
                          text=True,encoding='utf-8',capture_output=True)
    (folder/'new_controller_help.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    skill=(SCRIPTS.parent/'SKILL.md').read_text(encoding='utf-8')
    metadata=yaml.safe_load(skill.split('---',2)[1])
    valid=metadata.get('name')=='dpl-screening' and bool(metadata.get('description'))
    return dict(passed=result.returncode==0 and valid,syntax_files=files,
                target=str(TARGET),target_sha256=SOURCE_HASH,help_exit=result.returncode,format_valid=valid)



def r26(folder):
    work,ids=external_scores(folder,3)
    output=folder/'direct_selection'
    command=[sys.executable,'-B',str(SCRIPTS/'screen.py'),'--input-csv',str(at(work,'input.csv')),
       '--af3-scores',str(at(work,'scores/af3_scores.csv')),'--boltz2-scores',str(at(work,'scores/boltz2_scores.csv')),
       '--chai1-scores',str(at(work,'scores/chai1_scores.csv')),'--output-dir',str(output)]
    first=subprocess.run(command,cwd=OUT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
    p=at(work,'scores/af3_scores.csv');rows,cols=dpl.load_csv(p);rows[0]['score_mode']='mean5';rows[0]['af3_status']='OK_MEAN5';dpl.write_csv(p,rows,cols)
    second=subprocess.run(command,cwd=OUT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
    (folder/'direct_screen.txt').write_text(first.stdout+first.stderr+'\nSECOND\n'+second.stdout+second.stderr,encoding='utf-8')
    return dict(passed=first.returncode==0 and second.returncode!=0 and not output.exists(),
                first_exit=first.returncode,wrong_mode_exit=second.returncode,current_selection_removed=not output.exists())

def fake_probe(path, available):
    return dict(requested_python=str(path),python=str(path),version=[3,12,0],usable=True,
                modules={name:{'available':name in available} for name in dpl.MODULES})


def r27(folder):
    home=folder/'isolated_home';home.mkdir()
    source=folder/'ProteinA.fasta';source.write_text('>P00\nAC\n>P01\nAC\n',encoding='utf-8')
    cfg,path=configuration(folder)
    sim=PredictorSimulator(folder)
    with patch.object(dpl.Path,'home',return_value=home):
        code=invoke('all','--input',source,'--smiles','CCO','--config',path,sim=sim)
    work=home/'ProteinA_screening'
    expected={'af3','boltz2','chai1','run_info','selected_sequences.csv','selected_sequences.fasta'}
    assert {p.name for p in work.iterdir()}==expected
    for model,ext in [('af3','json'),('boltz2','yaml'),('chai1','fasta')]:
        assert {p.name for p in (work/model).iterdir()}=={'input','output'}
        assert (work/model/'input'/f'P00.{ext}').is_file()
        assert (work/model/'output/P00').is_dir()
        assert (work/model/'output'/f'{model}_scores.csv').is_file()
    for name in ('selected_sequences.csv','selected_sequences.fasta'):
        assert (work/name).read_bytes()==(work/'run_info/reports'/name).read_bytes()
    assert str(work/'selected_sequences.csv') in (work/'run_info/reports/screening_report.txt').read_text(encoding='utf-8')
    return dict(passed=code==0,default_workdir=str(work),exact_root_layout=True,
                full_pipeline_simulated=True,complete_sequences_in_root=True)


def r28(folder):
    cfg,path=configuration(folder)
    cfg['af3']['python']=str(folder/'af3_python')
    cfg['boltz2']['python']=str(folder/'boltz_python')
    cfg['chai1']['python']=sys.executable
    choices={cfg['af3']['python']:{'numpy'},cfg['boltz2']['python']:set(dpl.MODULES),sys.executable:set(dpl.MODULES)}
    with patch.object(dpl,'probe_python',side_effect=lambda p:fake_probe(p,choices[p])):
        report=dpl.environment_report(cfg)
        assert report['candidates'][0]['model']=='chai1'
        work=folder/'choice';dpl.init_workdir(work)
        plan=dpl.select_environment(cfg,work)
        assert set(plan['steps'].values())=={sys.executable}
        before=hashes(folder)
        js(path,cfg);before=hashes(folder)
        result=invoke('check-env','--config',path)
        unchanged=before==hashes(folder)
    return dict(passed=result==0 and unchanged,current_configured_environment_preferred=True,
                check_env_read_only=unchanged,installed_packages=0)


def r29(folder):
    cfg,path=configuration(folder)
    source=folder/'proteins.fasta';source.write_text('>P01\nAC\n',encoding='utf-8')
    work=folder/'approval'
    no_complete=lambda p:fake_probe(p,{'numpy','pandas'})
    with patch.object(dpl,'probe_python',side_effect=no_complete),patch.object(dpl,'make_processing_environment') as creation:
        status=invoke('check-env','--config',path)
        code=invoke('all','--input',source,'--smiles','CCO','--workdir',work,'--config',path)
        assert not creation.called
    state=read(work/'run_info/workflow_status.json')
    commands=[event for event in EVENTS if event.get('model') or any(x in str(event['argv']) for x in ('venv','pip'))]
    return dict(passed=status==2 and code==2 and state['status']=='approval_required' and not commands,
                check_exit=status,all_exit=code,workflow_state=state['status'],
                environment_created=(work/'run_info/processing_env').exists(),no_model_or_install_commands=not commands)


def r30(folder):
    cfg,path=configuration(folder)
    work=folder/'approved';dpl.init_workdir(work)
    partial=lambda p:fake_probe(p,{'numpy'})
    helper=work/'run_info/processing_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    complete=fake_probe(helper,dpl.MODULES)
    with patch.object(dpl,'probe_python',side_effect=partial),patch.object(dpl,'make_processing_environment',return_value=complete) as creation:
        plan=dpl.select_environment(cfg,work,'create')
        assert creation.call_count==1 and plan['mode']=='processing_env'
    with patch.object(dpl,'probe_python',side_effect=lambda p:complete if str(p)==str(helper) else partial(p)),patch.object(dpl,'make_processing_environment') as creation:
        resumed=dpl.select_environment(cfg,work,'auto')
        assert not creation.called and resumed['mode']=='processing_env'
    # Execute the actual command builder with mocked external venv/pip processes only.
    installer=folder/'installer';dpl.init_workdir(installer)
    calls=[]
    with patch.object(dpl,'run_cmd',side_effect=lambda cmd,*a,**k:calls.append(cmd)),patch.object(dpl,'probe_python',return_value=complete):
        dpl.make_processing_environment(installer,[partial(sys.executable)])
    expected_python=installer/'run_info/processing_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    assert calls[0][:3]==[sys.executable,'-m','venv']
    assert calls[1][:4]==[str(expected_python),'-m','pip','install']
    assert all('site-packages' not in str(cmd) for cmd in calls)
    failed=folder/'failed';dpl.init_workdir(failed)
    with patch.object(dpl,'run_cmd',side_effect=RuntimeError('simulated venv failure')):
        try:dpl.make_processing_environment(failed,[partial(sys.executable)])
        except RuntimeError:pass
        else:raise AssertionError('Failed venv creation must propagate')
    return dict(passed=True,explicit_create_selects_only_task_local_environment=True,
                approval_persists_for_resume=True,installer_commands=calls,real_installations=0,
                simulated_install_failure_did_not_modify_models=True)


def r31(folder):
    cfg,path=configuration(folder)
    for model in dpl.MODELS:cfg[model]['python']=str(folder/(model+'_python'))
    tools={'af3':{'rdkit.Chem','yaml'},'boltz2':{'numpy'},'chai1':{'numpy','pandas'}}
    lookup={cfg[m]['python']:tools[m] for m in dpl.MODELS}
    work=folder/'split';dpl.init_workdir(work)
    with patch.object(dpl,'probe_python',side_effect=lambda p:fake_probe(p,lookup[p])),patch.object(dpl,'make_processing_environment') as creation:
        plan=dpl.select_environment(cfg,work,'reuse')
        assert plan['mode']=='split' and not creation.called
        assert plan['steps']['normalize']==cfg['af3']['python']
        assert plan['steps']['reader_boltz2']==cfg['boltz2']['python']
        assert plan['steps']['reader_chai1']==cfg['chai1']['python']
        repeat=dpl.select_environment(cfg,work,'auto')
        assert repeat==plan
    missing=folder/'missing';dpl.init_workdir(missing)
    with patch.object(dpl,'probe_python',side_effect=lambda p:fake_probe(p,{'numpy'})),patch.object(dpl,'make_processing_environment') as creation:
        try:dpl.select_environment(cfg,missing,'reuse')
        except RuntimeError as exc:assert 'normalize' in str(exc)
        else:raise AssertionError('Missing RDKit must stop')
        assert not creation.called
    # The real built-in worker is a subprocess of the selected interpreter, not a generated script.
    command=[sys.executable,'-B',str(TARGET),'_cpu','smiles']
    result=subprocess.run(command,input=json.dumps({'smiles':'CCO'}),capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
    assert result.returncode==0 and json.loads(result.stdout) is True
    return dict(passed=True,selected_steps=plan['steps'],refusal_remembered=True,
                missing_step_stops_without_installation=True,actual_existing_script_worker_exit=result.returncode)


def r32(folder):
    work,source,cfg,path=prepared(folder,ids=['P01'])
    sim=PredictorSimulator(folder)
    original=sim.run
    def no_pocket(model,cmd,env):
        original(model,cmd,env)
        if model=='af3':
            # Put ligand far from every protein residue, keeping valid structures/confidences.
            for cif in Path(cmd[cmd.index('--output_dir')+1]).rglob('*model.cif'):
                text=cif.read_text(encoding='utf-8')
                for old,new in [('? 1 0 0 80','? 100 0 0 80'),('? 1.5 0 0 80','? 101 0 0 80'),('? 2 0 0 80','? 102 0 0 80')]:text=text.replace(old,new)
                cif.write_text(text,encoding='utf-8')
    sim.run=no_pocket
    a=invoke('predict','--workdir',work,'--config',path,sim=sim)
    b=invoke('collect','--workdir',work,'--config',path)
    c=invoke('screen','--workdir',work)
    rows,_=dpl.load_csv(work/'af3/output/af3_scores.csv')
    assert 'pocket_residue_plddt_unavailable' in rows[0]['af3_status']
    assert rows[0]['pocket_mean_plddt_5A']==''
    warnings,_=dpl.load_csv(work/'run_info/reports/score_warnings.csv')
    assert any(r['warning']=='pocket_residue_plddt_unavailable' for r in warnings)
    resume=PredictorSimulator(folder)
    d=invoke('predict','--workdir',work,'--config',path,sim=resume)
    assert not resume.launches
    return dict(passed=all(x==0 for x in (a,b,c,d)),exit_codes=[a,b,c,d],source_status_preserved=rows[0]['af3_status'],
                warning_recorded=True,missing_auxiliary_pocket_does_not_trigger_rerun=True)


def r33(folder):
    scenarios=['unknown_warning','missing_required_structure','boltz_required_pocket_missing','valid_af3_auxiliary']
    observations=[]
    for variant in scenarios:
        work,ids=external_scores(folder/variant,3)
        model='boltz2' if variant.startswith('boltz') else 'af3'
        p=dpl.score_path(work,model);rows,cols=dpl.load_csv(p)
        if variant=='unknown_warning':rows[0]['af3_status']='unknown_problem'
        elif variant=='missing_required_structure':rows[0]['af3_status']='missing_or_unreadable_model_cif'
        elif variant=='boltz_required_pocket_missing':rows[0][dpl.SCORE_RULES[model]['score']]=''
        else:rows[0]['af3_status']='pocket_residue_plddt_unavailable'
        dpl.write_csv(p,rows,cols)
        result=invoke('screen','--workdir',work,'--allow-unverified-scores')
        observations.append(dict(variant=variant,exit_code=result))
        assert (result==0)==(variant=='valid_af3_auxiliary')
    return dict(passed=True,observations=observations,unknown_and_required_failures_not_relaxed=True)


def r34(folder):
    work,ids=external_scores(folder,3)
    assert invoke('screen','--workdir',work,'--allow-unverified-scores')==0
    previous={name:(work/name).read_bytes() for name in ('selected_sequences.csv','selected_sequences.fasta')}
    original=dpl.os.replace
    def fail_second(source,dest):
        if Path(dest)==work/'selected_sequences.fasta':raise OSError('Injected second-export publication failure')
        return original(source,dest)
    with patch.object(dpl.os,'replace',side_effect=fail_second):
        result=invoke('screen','--workdir',work,'--allow-unverified-scores')
    assert result!=0 and not dpl.reports(work).exists()
    assert all(not (work/name).exists() for name in previous)
    for name,content in previous.items():
        assert any(p.read_bytes()==content for p in (work/'run_info/backups/published_sequences').glob(name+'_*'))
    assert read(work/'run_info/workflow_status.json')['status']=='failed'
    return dict(passed=True,second_export_failure_exit=result,no_partial_current_exports=True,previous_exports_archived=True)


def r35(folder):
    work,source,cfg,path,sim=pipeline(folder/'reserved',ids=['af3_scores.csv','batch_run_summary.tsv'])
    rows,_=dpl.load_csv(work/'selected_sequences.csv')
    assert {r['original_protein_id'] for r in rows}=={'af3_scores.csv','batch_run_summary.tsv'}
    old=folder/'old';old.mkdir();(old/'input.csv').write_text('Do not migrate',encoding='utf-8')
    before=hashes(old)
    result=invoke('normalize','--input',source,'--config',path,'--workdir',old)
    assert result!=0 and before==hashes(old)
    source2=folder/'dry.fasta';source2.write_text('>P01\nAC\n',encoding='utf-8')
    with patch.object(dpl,'make_processing_environment') as creation:
        dry=invoke('all','--input',source2,'--smiles','CCO','--config',path,'--workdir',folder/'dry','--dry-run','--env-policy','create')
        assert dry!=0 and not creation.called
    return dict(passed=True,output_filename_collision_aliased=True,old_layout_unchanged=True,dry_run_cannot_install=True)


def r36(folder):
    work,source,cfg,path=prepared(folder,ids=['P01'])
    assert invoke('predict','--workdir',work,'--config',path,sim=PredictorSimulator(folder))==0
    assert invoke('collect','--workdir',work,'--config',path)==0
    jobs=[('smiles',{'smiles':'CCO'}),('npz',{'path':str(work/'chai1/output/P01/seed_1/scores.model_idx_0.npz')}),
          ('arrays',{'model':'boltz2','directory':str(work/'boltz2/output/P01/predictions/P01'),'task':'P01'}),
          ('arrays',{'model':'chai1','directory':str(work/'chai1/output/P01/seed_1'),'task':'P01'})]
    outcomes=[]
    for operation,payload in jobs:
        result=subprocess.run([sys.executable,'-B',str(TARGET),'_cpu',operation],input=json.dumps(payload),
            capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
        assert result.returncode==0 and json.loads(result.stdout) is True,result.stderr
        outcomes.append(dict(operation=operation,model=payload.get('model'),exit_code=result.returncode))
    return dict(passed=True,actual_existing_script_cpu_subprocesses=outcomes,no_generated_worker_scripts=True)


def r37(folder):
    # A real task-local stdlib-only bootstrap verifies that the controller does not
    # accidentally rely on the interpreter that starts it having scientific tools.
    bootstrap=folder/'bootstrap'
    creation=subprocess.run([sys.executable,'-B','-m','venv','--without-pip',str(bootstrap)],
        capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
    assert creation.returncode==0,creation.stderr
    python=bootstrap/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    probe=subprocess.run([str(python),'-I','-B','-c',
        "import importlib.util,json;print(json.dumps([importlib.util.find_spec(x) is None for x in ('rdkit','numpy','pandas','yaml')]))"],
        capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
    assert probe.returncode==0 and all(json.loads(probe.stdout))
    cfg,path=configuration(folder)
    source=folder/'source.csv'
    dpl.write_csv(source,[dict(task_id='P01',Protein_seq='AC',ligand_smiles='CCO')],list(dpl.REQUIRED))
    work=folder/'run';commands=[]
    def external(*args):
        command=[str(python),'-I','-B',str(TARGET),*map(str,args)]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
        commands.append(dict(argv=command,exit_code=result.returncode))
        with (folder/'stdlib_bootstrap_execution.txt').open('a',encoding='utf-8') as stream:
            stream.write(repr(command)+'\n'+result.stdout+result.stderr+'\n')
        assert result.returncode==0,result.stderr
    external('normalize','--input',source,'--config',path,'--workdir',work)
    external('prepare','--workdir',work)
    assert invoke('predict','--workdir',work,'--config',path,sim=PredictorSimulator(folder))==0
    external('collect','--workdir',work,'--config',path)
    external('screen','--workdir',work)
    assert (work/'selected_sequences.fasta').is_file()
    return dict(passed=True,bootstrap_has_no_scientific_dependencies=True,
                actual_stdlib_only_controller_commands=commands,venv_created_without_pip=True,
                original_readers_executed_in_reused_environment=True,model_environments_modified=False)


def pre_affinity_arrays():
    chains=np.array([('A',0,2,0,2),('B',3,3,2,1)],dtype=[('name','U4'),('mol_type','i4'),
        ('atom_num','i4'),('res_idx','i4'),('res_num','i4')])
    residues=np.array([(1,),(1,),(3,)],dtype=[('atom_num','i4')])
    return dict(chains=chains,residues=residues,pocket=np.array(None,dtype=object))


def r38(folder):
    checks=[]
    for variant in ('object_none','without_optional_pocket','numeric_pocket'):
        arrays=pre_affinity_arrays()
        if variant=='without_optional_pocket':arrays.pop('pocket')
        if variant=='numeric_pocket':arrays['pocket']=np.array([0,1])
        path=folder/(variant+'.npz');np.savez(path,**arrays)
        accepted=dpl._valid_npz(path,profile='boltz_pre_affinity')
        assert accepted
        checks.append(dict(variant=variant,accepted=accepted))
    assert not dpl._valid_npz(folder/'object_none.npz')
    variants=('missing_chains','missing_residues','missing_chain_field','missing_residue_field',
              'unstructured_chains','unstructured_residues','empty_chains','empty_residues',
              'two_dimensional_chains','object_required_chains','object_other_member')
    for variant in variants:
        arrays=pre_affinity_arrays()
        if variant=='missing_chains':arrays.pop('chains')
        elif variant=='missing_residues':arrays.pop('residues')
        elif variant=='missing_chain_field':arrays['chains']=arrays['chains'][['name','mol_type','atom_num','res_idx']]
        elif variant=='missing_residue_field':arrays['residues']=np.ones(3,dtype=[('wrong','i4')])
        elif variant=='unstructured_chains':arrays['chains']=np.ones(2)
        elif variant=='unstructured_residues':arrays['residues']=np.ones(3)
        elif variant=='empty_chains':arrays['chains']=arrays['chains'][:0]
        elif variant=='empty_residues':arrays['residues']=arrays['residues'][:0]
        elif variant=='two_dimensional_chains':arrays['chains']=arrays['chains'].reshape(1,2)
        elif variant=='object_required_chains':arrays['chains']=np.array([None],dtype=object)
        else:arrays['unknown_object']=np.array(None,dtype=object)
        path=folder/(variant+'.npz');np.savez(path,**arrays)
        accepted=dpl._valid_npz(path,profile='boltz_pre_affinity')
        assert not accepted,variant
        checks.append(dict(variant=variant,accepted=accepted))
    return dict(passed=True,checks=checks,generic_numeric_policy_still_rejects_objects=True)


def r39(folder):
    base=folder/'valid.npz';np.savez(base,**pre_affinity_arrays())
    with zipfile.ZipFile(base) as archive:members={name:archive.read(name) for name in archive.namelist()}
    import numpy.lib.format as fmt
    import io as io_module
    stream=io_module.BytesIO(members['pocket.npy'])
    version=fmt.read_magic(stream)
    if version==(1,0):fmt.read_array_header_1_0(stream)
    else:fmt.read_array_header_2_0(stream)
    split=stream.tell();header=members['pocket.npy'][:split];payload=members['pocket.npy'][split:]
    checks=[]
    variants=('invalid_npy_payload','invalid_optional_header','invalid_optional_pickle',
              'truncated_optional_pickle','trailing_optional_pickle','truncated_zip','crc_corrupt_optional',
              'duplicate_member','zero_bytes','non_zip','missing_file')
    for variant in variants:
        path=folder/(variant+'.npz')
        if variant=='missing_file':pass
        elif variant=='zero_bytes':path.touch()
        elif variant=='non_zip':path.write_bytes(b'not a zip')
        elif variant=='truncated_zip':path.write_bytes(base.read_bytes()[:-16])
        elif variant=='crc_corrupt_optional':
            raw=bytearray(base.read_bytes())
            with zipfile.ZipFile(base) as archive:entry=archive.getinfo('pocket.npy')
            # np.savez uses stored members: flip a data byte but preserve the stored CRC.
            start=entry.header_offset
            name_len=int.from_bytes(raw[start+26:start+28],'little')
            extra_len=int.from_bytes(raw[start+28:start+30],'little')
            raw[start+30+name_len+extra_len+entry.file_size-2]^=1
            path.write_bytes(raw)
        else:
            changed=dict(members)
            if variant=='invalid_npy_payload':changed['chains.npy']=b'invalid NumPy header'
            elif variant=='invalid_optional_header':changed['pocket.npy']=b'invalid NumPy header'
            elif variant=='invalid_optional_pickle':changed['pocket.npy']=header+b'invalid pickle'
            elif variant=='truncated_optional_pickle':changed['pocket.npy']=header+payload[:-1]
            elif variant=='trailing_optional_pickle':changed['pocket.npy']=header+payload+b'extra'
            with zipfile.ZipFile(path,'w') as archive:
                for name,data in changed.items():archive.writestr(name,data)
                if variant=='duplicate_member':archive.writestr('chains.npy',members['chains.npy'])
        accepted=dpl._valid_npz(path,profile='boltz_pre_affinity')
        assert not accepted,variant
        checks.append(dict(variant=variant,accepted=accepted))
    return dict(passed=True,checks=checks,all_corruption_variants_rejected=True)


def r40(folder):
    work,source,cfg,path,sim=pipeline(folder,ids=['P01'])
    pre=work/'boltz2/output/P01/predictions/P01/pre_affinity_P01.npz'
    marker=folder/'UNPICKLING_MUST_NOT_RUN'
    class Trap:
        def __reduce__(self):return (Path.touch,(marker,))
    arrays=pre_affinity_arrays();arrays['pocket']=np.array(Trap(),dtype=object)
    trapped=folder/'unused_object.npz';np.savez(trapped,**arrays)
    assert dpl._valid_npz(trapped,profile='boltz_pre_affinity') and not marker.exists()
    # Original scientific reader never accesses this unused object field either.
    shutil.copy2(trapped,pre)
    row=dpl.score_reader('boltz2').collect_one_task(work/'boltz2/output/P01', 0, None, 5.0)
    dpl.selector().validate_model_score('boltz2',row,metadata=False)
    assert not marker.exists()
    assert dpl.completed_files('boltz2',work/'boltz2/output','P01')
    worker_checks=[]
    for profile,expected in [('boltz_pre_affinity',True),('numeric',False)]:
        command=[sys.executable,'-B',str(TARGET),'_cpu','npz']
        result=subprocess.run(command,input=json.dumps({'path':str(trapped),'profile':profile}),
            capture_output=True,text=True,encoding='utf-8',env=os.environ.copy())
        assert result.returncode==0 and json.loads(result.stdout) is expected,result.stderr
        assert not marker.exists()
        worker_checks.append(dict(profile=profile,accepted=json.loads(result.stdout)))
    return dict(passed=True,normal_pocket_none_pipeline_passes=True,unused_object_never_executed=True,
                original_reader_status=row['boltz2_status'],required_score=row['boltz2_pocket_mean_plddt_5A'],
                cpu_worker_checks=worker_checks,scientific_reader_unmodified=True)


def r41(folder):
    sample=os.environ.get('DPL_PRE_AFFINITY_SAMPLE')
    if not sample:
        return dict(passed=True,external_sample_provided=False,real_files_tested=0,
                    note='Set DPL_PRE_AFFINITY_SAMPLE to test a private real-model file without bundling it.')
    path=Path(sample).resolve();before=hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.is_file()
    generic=dpl._valid_npz(path)
    specialized=dpl._valid_npz(path,profile='boltz_pre_affinity')
    assert not generic and specialized
    with np.load(path,allow_pickle=False) as arrays:
        chains=arrays['chains'];residues=arrays['residues']
        chain_names=[str(x) for x in chains['name']]
        binder=next(str(c['name']) for c in chains if int(c['mol_type'])==3)
        tokens=sum(int(c['res_num']) if int(c['mol_type'])==0 else int(c['atom_num']) if str(c['name'])==binder or int(c['mol_type'])==3 else int(c['res_num']) for c in chains)
    reader=dpl.score_reader('boltz2')
    mapped=reader.chain_infos_from_pre_affinity(path,binder,tokens)
    assert len(mapped)==len(chains) and before==hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(passed=True,external_sample_provided=True,real_files_tested=1,sample_path=str(path),sample_sha256=before,
                old_generic_gate=generic,new_specialized_gate=specialized,chains=len(chains),residues=len(residues),
                original_reader_chain_mapping_succeeded=True,source_unchanged=True,
                full_real_prediction_bundle_available=False)


def r42(folder):
    # Instrument real stage boundaries while the model launches alone are simulated.
    work,source,cfg,path=prepared(folder,ids=IDS[:3])
    sim=PredictorSimulator(folder)
    timeline=[]
    original_dispatch=dispatch
    def observe(cmd,log=None,env=None,dry=False):
        child=Path(cmd[1])
        model=PREDICTORS.get(child.name)
        if model:
            timeline.append('start:'+model)
            result=original_dispatch(cmd,log,env,dry)
            timeline.append('end:'+model)
            return result
        if child.name=='score_to_csv.py':timeline.append('score:'+child.parent.name)
        if child==dpl.SCREEN:timeline.append('screen')
        return original_dispatch(cmd,log,env,dry)
    audit,check=dpl.audit_predictions,dpl.check_output
    def auditing(*args):
        timeline.append('audit')
        return audit(*args)
    def checking(model,root,key):
        timeline.append('check:'+model+'/'+key)
        return check(model,root,key)
    with (patch.object(sys.modules[__name__],'dispatch',side_effect=observe),
          patch.object(dpl,'audit_predictions',side_effect=auditing),
          patch.object(dpl,'check_output',side_effect=checking),
          patch.object(dpl.score_reader('af3'),'collect_one_job',side_effect=AssertionError('early AF3 extraction')),
          patch.object(dpl.score_reader('boltz2'),'collect_one_task',side_effect=AssertionError('early Boltz extraction')),
          patch.object(dpl.score_reader('chai1'),'collect_one_sample',side_effect=AssertionError('early Chai extraction'))):
        assert invoke('all','--config',path,'--input',source,'--workdir',work,sim=sim)==0
    expected=[edge+':'+m for m in dpl.MODELS for edge in ('start','end')]+['audit']
    expected+=['check:'+m+'/'+key for m in dpl.MODELS for key in IDS[:3]]
    expected+=['score:'+m for m in dpl.MODELS]+['screen']
    assert timeline==expected,timeline
    assert all(x['tasks']==IDS[:3] for x in sim.launches)
    return dict(passed=True,timeline=timeline,candidates_in_each_batch=3)


def r43(folder):
    import pathlib, types, datetime, textwrap, typing
    from unittest.mock import Mock
    keys=IDS[:3]
    # Execute the actual AF3 main + batch loop while substituting model and parser imports.
    path=SCRIPTS/'legacy/af3/run_alphafold.py'
    parsed=ast.parse(path.read_text(encoding='utf-8'))
    selected=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in ('main','run_batch_inputs')]
    inputs=folder/'inputs';inputs.mkdir()
    for key in keys:(inputs/(key+'.json')).touch()
    failures=folder/'af_errors.json'
    n=types.SimpleNamespace
    history=[]
    def infer(**kwargs):
        task=kwargs['fold_input'].name;history.append(task)
        if task==keys[1]:raise ValueError('synthetic AF3 failure')
    flags={item.value.id:None for node in selected for item in ast.walk(node)
           if isinstance(item,ast.Attribute) and item.attr=='value' and isinstance(item.value,ast.Name)}
    flags.update(_INPUT_DIR=str(inputs),_OUTPUT_DIR=str(folder/'afout'),_RUN_INFERENCE=True,_GPU_DEVICE=0,
                 _MAX_TEMPLATE_DATE='2020-01-01',MODEL_DIR=str(folder),_BUCKETS=[64])
    runner=Mock()
    env={**{k:n(value=v) for k,v in flags.items()},'os':os,'json':json,'pathlib':pathlib,
         'datetime':datetime,'textwrap':textwrap,'typing':typing,'tokamax':n(DotProductAttentionImplementation=str),
         'jax':n(local_devices=lambda **kw:[n(compute_capability='8.0')]),
         'ModelRunner':runner,'make_model_config':Mock(),'process_fold_input':infer,
         'folding_input':n(load_fold_inputs_from_path=lambda p:[n(name=p.stem)])}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
    with patch.dict(os.environ,{'DPL_BATCH_ERRORS':str(failures)}):
        try:env['main'](None)
        except SystemExit as exc:assert exc.code==1
    assert history==keys and runner.call_count==1 and set(read(failures))=={keys[1]}

    chai=load('tested_chai_batch',SCRIPTS/'legacy/chai1/run_chai_batch.py')
    failures=folder/'chai_errors.json';seen=[]
    def chai_infer(**kwargs):
        key=kwargs['fasta_file'].stem;seen.append(key)
        assert kwargs['seed']==1 and kwargs['num_diffn_samples']==1 and kwargs['use_esm_embeddings']
        if key==keys[1]:raise ValueError('synthetic Chai error')
    for key in keys:(inputs/(key+'.fasta')).touch()
    argv=['runner','-i',str(inputs),'-o',str(folder/'chaiout'),'--weights-dir',str(folder),'--num-results','1']
    with (patch.object(sys,'argv',argv),patch.dict(os.environ,{'DPL_BATCH_ERRORS':str(failures)}),
          patch.dict(sys.modules,{'chai_lab':n(),'chai_lab.chai1':n(run_inference=chai_infer),
                                  'torch':n(cuda=n(is_available=lambda:False))})):
        try:chai.main()
        except SystemExit as exc:assert exc.code==1
    assert seen==keys and set(read(failures))=={keys[1]}
    return dict(passed=True,af3_model_initializations=runner.call_count,af3_attempts=history,chai_attempts=seen,
                caught_middle_failure_both=True)


def r44(folder):
    work,source,cfg,path=prepared(folder,ids=IDS[:3])
    cfg['execution']['seconds_per_sample']={'af3':40,'boltz2':30,'chai1':20};js(path,cfg)
    sim=PredictorSimulator(folder)
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=sim)==0
    def batch(model):
        stamp=read(dpl.info(work)/'prediction_wave.json')['batches'][model]
        return read(dpl.info(work)/'batches'/stamp/'execution.json')
    assert [batch(m)['estimate_seconds'] for m in dpl.MODELS]==[120,90,60]
    history=read(dpl.info(work)/'timing_history.json')
    # A damaged prior output is reported at the barrier. Only its receipt is retried.
    corrupt=work/'chai1/output/P01/seed_1/pred.model_idx_0.cif';corrupt.write_bytes(b'corrupt')
    assert invoke('collect','--workdir',work,'--config',path)!=0
    second=PredictorSimulator(folder)
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=second)==0
    assert [(x['model'],x['tasks']) for x in second.launches]==[('chai1',['P01'])]
    assert batch('chai1')['estimate_seconds']==history['chai1']['seconds_per_sample']
    assert any(p.read_bytes()==b'corrupt' for p in (dpl.info(work)/'backups/predictions/chai1').rglob('*.cif'))
    return dict(passed=True,initial_seconds=[120,90,60],retry_seconds=batch('chai1')['estimate_seconds'],
                retry_candidates=second.launches[0]['tasks'],preserved_previous_prediction=True)


def r45(folder):
    work,source,cfg,path=prepared(folder,ids=IDS[:3])
    sim=PredictorSimulator(folder,omit_af3_conf=True)
    original=sim.run
    def mixed(model,cmd,env):
        original(model,cmd,env)
        if model=='boltz2':
            out=Path(cmd[cmd.index('-o')+1])
            p=out/'boltz_results_i/predictions/P01/plddt_P01_model_0.npz';p.unlink()
            raise subprocess.CalledProcessError(1,cmd)
        if model=='chai1':
            out=Path(cmd[cmd.index('-o')+1])
            np.savez(out/'P02/seed_1/scores.model_idx_0.npz',unrelated=np.ones(3))
    sim.run=mixed
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=sim)!=0
    failures=read(dpl.info(work)/'prediction_audit.json')['errors']
    assert set(failures)=={'af3/P00','af3/P01','af3/P02','boltz2/P01','chai1/P02'},failures
    assert not any('score_to_csv.py' in ' '.join(x['argv']) for x in EVENTS)
    assert all(not (work/name).exists() for name in ('selected_sequences.csv','selected_sequences.fasta'))
    assert dpl.trusted_complete('boltz2',work,cfg,'P00') and dpl.trusted_complete('boltz2',work,cfg,'P02')
    recovery=PredictorSimulator(folder)
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=recovery)==0
    assert [x['tasks'] for x in recovery.launches]==[IDS[:3],['P01'],['P02']]
    return dict(passed=True,all_failures=failures,recovery_batches=[x['tasks'] for x in recovery.launches])


def r46(folder):
    work,source,cfg,path=prepared(folder,ids=IDS[:2])
    sim=PredictorSimulator(folder)
    original=sim.run
    def native_exit(model,cmd,env):
        original(model,cmd,env)
        if model=='boltz2':raise subprocess.CalledProcessError(7,cmd)
    sim.run=native_exit
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=sim)!=0
    assert [x['model'] for x in sim.launches]==list(dpl.MODELS)
    assert set(read(dpl.info(work)/'prediction_audit.json')['errors'])=={'boltz2/P00','boltz2/P01'}
    assert not dpl.score_path(work,'af3').exists()
    recover=PredictorSimulator(folder)
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=recover)==0
    assert [x['model'] for x in recover.launches]==['boltz2']
    # Another controller must fail before changing receipts, outputs or current results.
    before=hashes(dpl.reports(work))
    with dpl.workflow_lock(work):
        assert invoke('predict','--config',path,'--workdir',work,sim=PredictorSimulator(folder))!=0
    assert before==hashes(dpl.reports(work))
    return dict(passed=True,unattributed_nonzero_blocked=True,subsequent_model_ran=True,concurrent_writer_excluded=True)


def r47(folder):
    work,source,cfg,path=prepared(folder,ids=IDS[:2])
    prior=folder/'previous.json'
    js(prior,{m:{'seconds_per_sample':i+1,'hardware':dpl.recipe_hash(cfg[m])} for i,m in enumerate(dpl.MODELS)})
    cfg['execution']['history_file']=prior.name;js(path,cfg)
    assert invoke('predict','--workdir',work,'--config',path,sim=PredictorSimulator(folder))==0
    wave=read(dpl.info(work)/'prediction_wave.json')
    times=[read(dpl.info(work)/'batches'/wave['batches'][m]/'execution.json')['estimate_seconds'] for m in dpl.MODELS]
    assert times==[2,4,6]
    batch=dpl.info(work)/'batches'/wave['batches']['af3']
    (batch/'i/P01.json').write_text('{}',encoding='utf-8')
    assert invoke('collect','--workdir',work,'--config',path)!=0
    assert set(read(dpl.info(work)/'prediction_audit.json')['errors'])=={'af3/P01'}
    resume=PredictorSimulator(folder)
    assert invoke('all','--input',source,'--config',path,'--workdir',work,sim=resume)==0
    assert resume.launches[0]['tasks']==['P01'] and len(resume.launches)==1
    record=dpl.info(work)/'batches'/wave['batches']['chai1']/'execution.json'
    data=read(record);data['exit_code']=9;js(record,data)
    assert invoke('screen','--workdir',work)!=0
    return dict(passed=True,imported_timing_seconds=times,staged_input_change_detected=True,changed_batch_record_blocked=True)


def r48(folder):
    # Let the real subprocess.run wait on three short-lived OS processes. Each
    # child emits process-boundary events itself, and produces no predictions.
    work,source,cfg,path=prepared(folder,ids=IDS[:3])
    trace=folder/'children.jsonl'
    def cpu_command(model,settings,stage,outdir,errors):
        code=(
            'import json,os,sys,time;from pathlib import Path\n'
            'model,stage,trace=sys.argv[1:]\n'
            'for event in ("start","end"):\n'
            '    with open(trace,"a",encoding="utf-8") as f:\n'
            '        print(json.dumps([event,model,os.getpid(),time.monotonic(),len(list(Path(stage).iterdir()))]),file=f)\n'
            '    if event=="start":time.sleep(0.05)\n')
        return [sys.executable,'-I','-B','-c',code,model,str(stage),str(trace)],os.environ.copy()
    with (patch.object(dpl,'batch_command',side_effect=cpu_command),
          patch.object(sys,'argv',[str(TARGET),'predict','--config',str(path),'--workdir',str(work)]),
          patch.object(dpl,'check_output',side_effect=AssertionError('prediction inspected during wait'))):
        assert dpl.main()==0
    events=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
    assert [(e[0],e[1]) for e in events]==[(edge,m) for m in dpl.MODELS for edge in ('start','end')]
    assert len({e[2] for e in events})==3 and all(e[4]==3 for e in events)
    assert all(events[i+1][3]>=events[i][3] for i in range(5))
    assert not any(dpl.score_path(work,m).exists() for m in dpl.MODELS)
    return dict(passed=True,child_events=events,blocking_wait_on_real_cpu_children=True)


ENTRIES=[
 ('R01_normal_pipeline',[],'Normal pipeline with real score extraction and controller-created receipts',r01),
 ('R02_repeat_and_resume',['F02'],'Repeated prepare/all dry-run, simulation and trusted resume',r02),
 ('R03_wrong_origin',['F01'],'Reject copied wrong-source predictions and receipts',r03),
 ('R04_prepared_tamper',['F04'],'Reject changed prepared sequence and seed',r04),
 ('R05_receipt_output_tamper',['F01','F05'],'Reject changed outputs, receipts and runtime settings',r05),
 ('R06_structure_validation',['F05'],'Empty/corrupt original repro plus invalid NPY payload',r06),
 ('R07_recovery_validation',['F05'],'Missing AF3 confidences and invalid NPZ dimensions across retries',r07),
 ('R08_failed_screen_archive',['F06'],'Early screen/all failure archives previous results',r08),
 ('R09_id_roundtrip',['F03'],'Numeric, NA, scientific-notation and punctuation ID roundtrip',r09),
 ('R10_fasta_txt',['F09'],'No dropped unnamed FASTA/TXT records',r10),
 ('R11_score_qc',['F07','F08'],'Reject partial, mean5, wrong sample and invalid scores',r11),
 ('R12_external_scores',['F01','F08'],'Default provenance enforcement and explicit external-score option',r12),
 ('R13_preflight',['F10'],'All-model preflight with missing paths and wrong path types',r13),
 ('R14_failed_batch_resume',['F01','F02','F11'],'Preserve successful jobs and evidence after a failed simulated batch',r14),
 ('R15_original_runner_keep_tmp',['F11'],'Original Boltz runner keeps synthetic failure evidence with flag',r15),
 ('R16_id_policy',['F13'],'Preserve original ID metadata and safe ID spelling',r16),
 ('R17_math_and_exports',[],'Known union/tie cases and 220-aa export',r17),
 ('R18_manifest_missing_receipts',['F01'],'Require provenance map even when a score manifest exists',r18),
 ('R19_collect_dry_run',[],'Dry-run must preserve a valid score manifest',r19),
 ('R20_failed_collect_status',['F06'],'Standalone collect failure must not leave current success state',r20),
 ('R21_skill_integration',['F12'],'Skill references and standalone package resources',r21),
 ('R22_required_rdkit',['F12'],'Real RDKit and declared dependency compatibility',r22),
 ('R23_invalid_inputs',[],'Invalid molecules, sequences, aliases and headers',r23),
 ('R24_parallel_resume',[],'Reject parallel mode and resume after GPU-only change',r24),
 ('R25_syntax_and_format',[],'Latest-controller CLI, syntax and skill manifest validation',r25),
 ('R26_direct_selector',[],'Direct selector strict single-mode gate and failed-result archival',r26),
 ('R27_home_layout',[],'Actual HOME default and exact public model/output layout',r27),
 ('R28_environment_choice',[],'Prefer complete configured current environment; read-only check-env',r28),
 ('R29_environment_approval',[],'No complete environment stops for approval without installing or predicting',r29),
 ('R30_approved_environment',[],'Explicit task-local creation, scope, persistence and installation failure',r30),
 ('R31_split_environment',[],'Refusal enables per-step reuse; missing tools stop; actual internal worker',r31),
 ('R32_af3_auxiliary_pipeline',[],'AF3 missing unused 5A confidence passes prediction/collection/screen/resume',r32),
 ('R33_required_score_gate',[],'Unknown statuses, required structures and mandatory Boltz pocket still fail',r33),
 ('R34_export_publication',[],'Second root export failure archives all current/previous result files',r34),
 ('R35_layout_safety',[],'Output-name aliases, old-workdir refusal and dry-run installation guard',r35),
 ('R36_cpu_workers',[],'Actual subprocess CPU workers for SMILES and array schema validation',r36),
 ('R37_stdlib_bootstrap',[],'Actual stdlib-only bootstrap delegates all CPU processing to an existing environment',r37),
 ('R38_pre_affinity_schema',[],'Optional pocket=None plus required chains/residues schema and strict numeric defaults',r38),
 ('R39_pre_affinity_corruption',[],'Reject CRC, ZIP, NPY, pickle syntax and required-array corruption',r39),
 ('R40_pre_affinity_pipeline',[],'Full None-pocket pipeline, original reader, CPU worker and no unused unpickling',r40),
 ('R42_batch_order',[],'Boundaries and exact call counts in all',r42),
 ('R43_real_runner_loops',[],'AF3 and Chai execution code with CPU framework doubles',r43),
 ('R44_estimate_and_resume',[],'Server timing estimate, targeted resume and backup',r44),
 ('R45_multi_model_errors',[],'Aggregate all errors and recover native partial success',r45),
 ('R46_fatal_and_lock',[],'Unattributed native exit and concurrent controller protection',r46),
 ('R47_history_and_origin',[],'Prior-run timing, staged input and batch identity',r47),
 ('R48_blocking_children',[],'Real blocking waits with three CPU child processes',r48),
 ('R41_real_pre_affinity',[],'Optional private real-model pre-affinity sample and original chain mapping',r41),
]


def main():
    selected=set(sys.argv[1:])
    if selected and (OUT/'results.json').is_file():
        RESULTS.extend(x for x in read(OUT/'results.json')['cases'] if x['id'] not in selected)
    for ident,related,title,fn in ENTRIES:
        if not selected or ident in selected:trial(ident,related,title,fn)
    RESULTS.sort(key=lambda x:x['id'])
    summary=dict(target=str(TARGET),target_sha256=SOURCE_HASH,cases=RESULTS,
        executed=sum(x['executed'] for x in RESULTS),harness_errors=sum(not x['executed'] for x in RESULTS),
        checks_meeting_full_expectation=sum(x['executed'] and x.get('detail',{}).get('passed') is True for x in RESULTS),
        real_gpu_jobs=0)
    js(OUT/'results.json',summary)
    print('GROUPS',len(RESULTS),'HARNESS_ERRORS',summary['harness_errors'])
    return 1 if summary['checks_meeting_full_expectation'] != len(RESULTS) else 0


if __name__=='__main__':raise SystemExit(main())
