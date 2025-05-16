import lmdb
import os
import pickle
from scipy.stats import rankdata
from utils.nmrnet import single_get_equi_class

class RetrievalLMDBDataset:
    def __init__(self, data_path: str, nmr_types: list):
        """
        初始化 LMDB 数据集。

        :param data_path: 数据文件路径（仅支持 lmdb 格式）。
        """
        self.data_path = data_path
        self.nmr_types = nmr_types

        self.data = self.load_data()
        self.query_data, self.library_data = self.process_data()
        
        self.num_query = len(self.query_data)
        self.num_library = len(self.library_data)

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件路径不存在: {self.data_path}")
        
        env = lmdb.open(
            self.data_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        txn_read = env.begin(write=False)

        data = {}
        with txn_read.cursor() as cursor:
            for key, value in cursor:
                key = key.decode('utf-8')
                data[key] = pickle.loads(value)

        env.close()

        return data
    
    def process_data(self):
        name_map = {'H': '1h', 'C': '13c'}
        query_data = {}
        library_data = {}
        
        for key, value in self.data.items():
            library_data[key] = {'atoms_element': value["atoms_element"],
                                 'atoms_pred': value["atoms_predict"],
                                 'mol': value["mol"],
                                 'atoms_equi_oracle': rankdata(value["atoms_target"], method='min'),
                                 'atoms_equi_topo': single_get_equi_class(value["mol"]),
                                 'smiles': value["smiles"]}
        
        for key, value in self.data.items():
            query_data_item = {nmr_type: value[f"nmr_{name_map[nmr_type]}"] for nmr_type in self.nmr_types}
            complete = {nmr_type: value.get(f"complete_{name_map[nmr_type]}", True) for nmr_type in self.nmr_types}
            if any(len(item) == 0 for item in query_data_item.values()):
                continue
            if any(item == False for item in complete.values()):
                continue
            query_data[key] = query_data_item

        return query_data, library_data
