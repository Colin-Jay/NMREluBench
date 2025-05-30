export PROJECT_ROOT="/fs_mol/fjxu/Spectra/NMR/NMREluBench/nmr_denovo"
export HYDRA_JOBS="/fs_mol/fjxu/Spectra/NMR/NMREluBench/nmr_denovo/hydra"
export WANDB_DIR="/fs_mol/fjxu/Spectra/NMR/NMREluBench/nmr_denovo/wandb"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 

[ -z "${expname}" ] && expname="test"

[ -z "${data_name}" ] && data_name="qm9"
[ -z "${batch_size}" ] && batch_size=32
[ -z "${use_h}" ] && use_h=False
[ -z "${use_c}" ] && use_c=True
[ -z "${use_complete}" ] && use_complete=False

[ -z "${ckpt_path}" ] && ckpt_path=null
[ -z "${test_result_path}" ] && test_result_path=null
[ -z "${model_encoder}" ] && model_encoder=mlp
[ -z "${model_decoder}" ] && model_decoder=bart
[ -z "${model_tokenizer}" ] && model_tokenizer=smiles

[ -z "${lr}" ] && lr=1e-4

[ -z "${max_epochs}" ] && max_epochs=100
[ -z "${test_only}" ] && test_only=True


python nmr_run.py expname=$expname \
    data=$data_name \
    data.batch_size=$batch_size  \
    data.datamodule.use_h=$use_h  \
    data.datamodule.use_c=$use_c  \
    data.datamodule.use_complete=$use_complete \
    model=denovo \
    model.ckpt_path=$ckpt_path \
    model.test_result_path=$test_result_path \
    model/encoder=$model_encoder \
    model/decoder=$model_decoder \
    model/tokenizer=$model_tokenizer \
    test_only=$test_only \

