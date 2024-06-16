exp_ini=$1

python LNCHTM.py $exp_ini
rm ../result/$exp_ini/chunk_0/res.json
python evaluation.py $exp_ini