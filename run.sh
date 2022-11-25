cp -r /code/* /workspace && cp /dataset/* /workspace/data && cd /workspace
python3 preprocess.py
python3 data_augment.py
python3 -u supervised.py --n_gpu 2