from pathlib import Path
import json
import numpy as np
import yaml

def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

CIF = "data_synthetic\nloop_\n_entity.id\n_entity.type\n1 polymer\n2 non-polymer\n#\nloop_\n_entity_poly.entity_id\n_entity_poly.type\n1 'polypeptide(L)'\n#\nloop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\nB 2\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_entity_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\nATOM 1 C CA ALA A 1 1 1 ? 0 0 0 80\nATOM 2 C CA CYS A 1 2 2 ? 3 0 0 80\nHETATM 3 C C1 LIG B 2 . 1 ? 1 0 0 80\nHETATM 4 C C2 LIG B 2 . 1 ? 1.5 0 0 80\nHETATM 5 O O1 LIG B 2 . 1 ? 2 0 0 80\n#\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n_ma_qa_metric.type\n1 local pLDDT pLDDT\n#\nloop_\n_ma_qa_metric_local.ordinal_id\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n1 A 1 ALA 1 80\n2 A 2 CYS 1 80\n3 B . LIG 1 80\n#\n"

def pdb_line(record, number, atom, res, chain, resid, x, element):
    return f'{record:<6}{number:5d} {atom:^4} {res:>3} {chain}{resid:4d}    {x:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{80:6.2f}          {element:>2}\n'

def predictions(work, task, values=(0.8, 0.8, 0.8)):
    a, b, c = values
    root = work / 'predictions/af3' / task
    for i in range(5):
        p = root / f'seed-1_sample-{i}'
        p.mkdir(parents=True, exist_ok=True)
        (p / f'{task}_model.cif').write_text(CIF, encoding='utf-8')
        js(p / f'{task}_summary_confidences.json', dict(ranking_score=a if i == 0 else 0.999, iptm=0.7, ptm=0.7, fraction_disordered=0, has_clash=False, chain_ids=['A', 'B'], chain_pair_iptm=[[0.7, 0.6], [0.5, 0.7]]))
        js(p / f'{task}_confidences.json', dict(atom_chain_ids=['A', 'A', 'B', 'B', 'B'], atom_plddts=[80] * 5))
    js(root / f'{task}_data.json', dict(name=task, modelSeeds=[1], sequences=[{'protein': {'sequence': 'AC'}}, {'ligand': {'smiles': 'CCO'}}]))
    p = work / 'predictions/boltz2' / task / 'predictions' / task
    p.mkdir(parents=True, exist_ok=True)
    pdb = ''.join([pdb_line('ATOM', 1, 'CA', 'ALA', 'A', 1, 0, 'C'), pdb_line('ATOM', 2, 'CA', 'CYS', 'A', 2, 3, 'C'), pdb_line('HETATM', 3, 'C1', 'LIG', 'B', 1, 1, 'C'), pdb_line('HETATM', 4, 'C2', 'LIG', 'B', 1, 1.5, 'C'), pdb_line('HETATM', 5, 'O1', 'LIG', 'B', 1, 2, 'O')]) + 'END\n'
    (p / f'{task}_model_0.pdb').write_text(pdb, encoding='utf-8')
    confidence = {k: 0.7 for k in ('confidence_score', 'ptm', 'iptm', 'ligand_iptm', 'protein_iptm', 'complex_plddt', 'complex_iplddt', 'complex_pde', 'complex_ipde')}
    confidence.update(chains_ptm={'0': 0.7, '1': 0.7}, pair_chains_iptm={'0': {'0': 0.7, '1': 0.6}, '1': {'0': 0.5, '1': 0.7}})
    js(p / f'confidence_{task}_model_0.json', confidence)
    js(p / f'affinity_{task}.json', dict(affinity_pred_value=0, affinity_probability_binary=0.5, affinity_pred_value1=0, affinity_probability_binary1=0.5, affinity_pred_value2=0, affinity_probability_binary2=0.5))
    np.savez(p / f'plddt_{task}_model_0.npz', plddt=np.array([b, b, 0.8, 0.8, 0.8]))
    np.savez(p / f'pae_{task}_model_0.npz', pae=np.ones((5, 5)))
    np.savez(p / f'pde_{task}_model_0.npz', pde=np.ones((5, 5)))
    chains = np.array([('A', 0, 2, 0, 2), ('B', 3, 3, 2, 1)], dtype=[('name', 'U4'), ('mol_type', 'i4'), ('atom_num', 'i4'), ('res_idx', 'i4'), ('res_num', 'i4')])
    residues = np.array([(1,), (1,), (3,)], dtype=[('atom_num', 'i4')])
    np.savez(p / f'pre_affinity_{task}.npz', chains=chains, residues=residues, pocket=None)
    (p.parents[1] / 'input_used.yaml').write_text(yaml.safe_dump(dict(version=1, sequences=[{'protein': {'id': 'A', 'sequence': 'AC', 'msa': 'empty'}}, {'ligand': {'id': 'B', 'smiles': 'CCO'}}], properties=[{'affinity': {'binder': 'B'}}])), encoding='utf-8')
    p = work / 'predictions/chai1' / task / 'seed_1'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'pred.model_idx_0.cif').write_text(CIF, encoding='utf-8')
    np.savez(p / 'scores.model_idx_0.npz', per_chain_pair_iptm=np.array([[0.7, c], [0.123, 0.7]]), aggregate_score=np.array(1 - c), ptm=np.array(0.7), iptm=np.array(0.7))
