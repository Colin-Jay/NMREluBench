from src.data import RetrievalLMDBDataset
from src.method import WassersteinMethod, VectorMethod, RandomMethod, CReSSMethod
from src.evaluate import NMRRetrievalEvaluator
import pandas as pd

name = "nmrshiftdb2_2024_nmrnet"
dataset = RetrievalLMDBDataset(f"./data/{name}/valid.lmdb", nmr_types=['C'])
print("num_query:", dataset.num_query)
print("num_library:", dataset.num_library)
evaluator = NMRRetrievalEvaluator(k_list=[1,2,3,4,5,6,7,8,9,10], metric_list=["recall", "MCES"]) # recall, MCES

results = {}
for method in [
    RandomMethod(seed=42),
    WassersteinMethod(weight_H=1, weight_C=0.1, equi_method='no'),
    WassersteinMethod(weight_H=1, weight_C=0.1, equi_method='topo'),
    WassersteinMethod(weight_H=1, weight_C=0.1, equi_method='oracle'),
    VectorMethod(dim=128, equi_method='no'),
    VectorMethod(dim=128, equi_method='topo'),
    VectorMethod(dim=128, equi_method='oracle'),
    CReSSMethod(
        config_path="./models/2_5_w_model/8.json",
        pretrain_model_path="./models/2_5_w_model/8.pth",
        device="cuda:0", # or "cpu"
    )
    ]:
    results[method.name] = evaluator.evaluate(dataset, method)

df_results = pd.DataFrame.from_dict(results, orient='index')

metrics_to_expand = {metric: pd.DataFrame(df_results[metric].tolist(), 
                                          columns=[f"{metric}@{k}" for k in evaluator.k_list], 
                                          index=df_results.index)
                     for metric in evaluator.metric_list}
expanded_results = pd.concat(metrics_to_expand.values(), axis=1)
df_results = pd.concat([df_results.drop(columns=metrics_to_expand.keys()), expanded_results], axis=1)

# save
df_results.to_csv("./results_c.csv", index_label="Method")

print('='*20 + ' Results ' + '='*20)
print(df_results)