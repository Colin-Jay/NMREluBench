from unimol_tools import MolPredict
from utils.lmdb import load_lmdb, write_lmdb
from utils.nmrnet import get_nmr_from_mol
from tqdm import tqdm
import os

split = 'valid'
dataset = 'nmrshiftdb2_2024'
base_model = f"./models/nmrnet"
path = f"./data/{dataset}/{split}.lmdb"

data = load_lmdb(path)

mol_list = [data[key]['mol'] for key in data]

result = get_nmr_from_mol(mol_list, clf=MolPredict(base_model), raw=True)
shift_list = result['shift_list']
element_list = result['element_list']

for i, key in tqdm(enumerate(data)):
    item = data[key]
    item['atoms_predict'] = shift_list[i]
    item['atoms_element'] = element_list[i]

new_path = os.path.join(
    os.path.dirname(path) + "_nmrnet",
    os.path.basename(path)
)
os.makedirs(os.path.dirname(new_path), exist_ok=True)
write_lmdb(data, new_path)
