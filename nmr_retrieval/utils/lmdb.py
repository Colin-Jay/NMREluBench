import lmdb
from tqdm import tqdm
import os
import pickle

def load_lmdb(path):
    # create an lmdb environmnet
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    # create a context for reading
    txn_read = env.begin(write=False)

    # read as key-value manner
    data = {}
    with txn_read.cursor() as cursor:
        for key, value in cursor:
            data[key] = pickle.loads(value)

    # close
    env.close()

    return data


def write_lmdb(data, path):

    try:
        os.remove(path)
        print("Remove existing lmdb: {}".format(os.path.abspath(path)))
    except:
        pass
    env_new = lmdb.open(
        path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        # max_readers=1,
        map_size=int(1e12),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    if isinstance(data, list):
        for index in tqdm(range(len(data))):
            inner_output = pickle.dumps(data[index], protocol=-1)
            txn_write.put(f'{i}'.encode("ascii"), inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    elif isinstance(data, dict):
        for key in tqdm(data.keys()):
            inner_output = pickle.dumps(data[key], protocol=-1)
            txn_write.put(key, inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    else:
        raise ValueError("Data type not supported: {}".format(type(data)))
    
    print("Write to lmdb: {}".format(os.path.abspath(path)))