import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from tqdm import tqdm
from multiprocessing import Pool
from unimol_tools import MolPredict

from unimol_tools.utils import logger
import logging
logger.setLevel(logging.ERROR)


def random_split(dataset, random_ratio=0.8):
    if random_ratio == None:
        return dataset, dataset
    else:
        key1 = list(dataset.keys())[0]
        list_length = len(dataset[key1])
        indices = list(range(list_length))
        random.shuffle(indices)
        train_split_index = int(random_ratio * list_length)
        random_set1 = {}
        random_set2 = {}
        for key in dataset:
            random_set1[key] = [dataset[key][i] for i in indices[:train_split_index]]
            random_set2[key] = [dataset[key][i] for i in indices[train_split_index:]]

        return random_set1, random_set2

def filter_data(molecules_nmrdata, filter_list, datatype, sampling_ratio=None):
    filtered_indices = []
    # print("sampling_ratio1", sampling_ratio)
    if 'all' in filter_list:
        if sampling_ratio==None:
            return molecules_nmrdata
        else:
            molecules_nmrdata, molecules_nmrdata2 = random_split(molecules_nmrdata, sampling_ratio)
            return molecules_nmrdata

    for i, atom_mask in enumerate(molecules_nmrdata['atom_mask']):
        if all(x == 0 for x in atom_mask):
            continue
        if any(x in filter_list for x in atom_mask):
            filtered_indices.append(i)
            
    filtered_dict = {}
    for key in molecules_nmrdata.keys():
        filtered_dict[key] = [molecules_nmrdata[key][i] for i in filtered_indices]

    filtered_dict, filtered_dict2 = random_split(filtered_dict, sampling_ratio)

    return filtered_dict

def map_element(nmrtype):
    result = []
    if nmrtype == 'ALL':
        result.append('all')
        return result
    for element in nmrtype.split("+"):
        result.append(Chem.GetPeriodicTable().GetAtomicNumber(element))

    return result

def get_atomic_numbers(mol):
    atomic_numbers = []  
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atomic_numbers.append(atomic_number)
    return atomic_numbers

def get_nmr_from_nmrnet(mol_list, clf, merge=False):
    infer_data = {
        'mol': [],
        'atom_target': [],
        'atom_mask': [],
    }
    for i, mol in enumerate(mol_list):

        infer_data['mol'].append(mol)
        infer_data['atom_target'].append([0.0]*512)
        atom_num = get_atomic_numbers(mol)
        infer_data['atom_mask'].append([0]+atom_num+[0]*(512-1-len(atom_num)))

    nmrtype = 'ALL'
    datatype='mol'
    filter_list = map_element(nmrtype)

    filtered_data = infer_data
    filtered_data = filter_data(filtered_data, filter_list, datatype)
    
    # 模型预测
    test_pred = clf.predict(filtered_data, datatype = datatype)
    target = clf.cv_true[clf.cv_label_mask]
    predict = clf.cv_pred[clf.cv_label_mask]
    index_mask = np.array(clf.cv_index_mask.tolist()).astype(np.int8)
    data_dict = {
        # 'cv_true': clf.cv_true,
        'cv_pred': clf.cv_pred,
        # 'cv_pred_fold': clf.cv_pred_fold,
        'cv_label_mask': clf.cv_label_mask,
        'index_mask': index_mask
    }
    if not merge:
        return data_dict

    nmr_list = []
    index_list = []
    for i in range(len(data_dict['index_mask'])):
        cv_pred=data_dict['cv_pred'][i].astype(np.float32)
        cv_label_mask=data_dict['cv_label_mask'][i]
        index_mask=data_dict['index_mask'][i]
        nmr_predict=cv_pred[cv_label_mask]
        mol_index=index_mask[cv_label_mask]
        nmr_list.append(nmr_predict)
        index_list.append(mol_index)
    return nmr_list, index_list


def get_equi_class(mol_list):
    with Pool() as pool:
        result = list(tqdm(pool.imap(single_get_equi_class, mol_list), total=len(mol_list)))
    return result

def single_get_equi_class(mol):
    equi_class = rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
    return np.array(equi_class).astype(np.int16)

def merge_equi_nmr(nmr, atom_index, equi_class, element_id=6):
    mask = atom_index == element_id
    equi_class = equi_class[mask]
    nmr = nmr[mask]
    unique_equi_class = np.unique(equi_class)
    # 获取每个 symmetry_class 的平均 atom_value
    return np.array([np.mean(nmr[equi_class == _class]) for _class in unique_equi_class])


def get_nmr_from_mol(mol_list, clf, elements=['C','H'], raw=False):
    if isinstance(mol_list, Chem.Mol):
        mol_list = [mol_list]
        
    if isinstance(mol_list[0], str):
        mol_list = [Chem.MolFromSmiles(s) for s in mol_list]
        mol_list = [Chem.AddHs(mol) for mol in mol_list]
        
    nmr_list, index_list = get_nmr_from_nmrnet(mol_list, clf, merge=True)
    equi_class_list = get_equi_class(mol_list)
    
    if raw:
        return {
            'mol_list': mol_list,
            'shift_list': nmr_list,
            'element_list': index_list,
            'equi_class_list': equi_class_list
        }

    H_nmr = []
    C_nmr = []
    for nmr, index, equi_class in zip(nmr_list, index_list, equi_class_list):
        if 'H' in elements:
            H_nmr.append(np.sort(nmr[index == 1]))
        if 'C' in elements:
            C_nmr.append(np.sort(merge_equi_nmr(nmr, index, equi_class, element_id=6)))

    return {'H_nmr': H_nmr, 'C_nmr': C_nmr}


def get_nmr_lists(shifts_list, elements_list, equi_classes_list, sorted=True):
    H_nmr_list = [shifts[elements == 1] for shifts, elements in zip(shifts_list, elements_list)]
    if sorted:
        H_nmr_list = [np.sort(H_nmr) for H_nmr in H_nmr_list]
    C_nmr_list = [np.sort(merge_equi_nmr(shifts, elements, equi_class, element_id=6)) for shifts, elements, equi_class in zip(shifts_list, elements_list, equi_classes_list)]
    return H_nmr_list, C_nmr_list
