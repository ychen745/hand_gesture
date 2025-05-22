#############
This is a new adaptation for ST-GCN hand gesture recognition. 
#############
1. Tested at 'Python 3.6.8/Pytorch 1.4.0’ and 'Python 3.8/3.7 and Pytorch 1.6.0'

2. Additional packages may need to be installed.

3. A quick test on toyDataset: 
python train.py -c config/train_toyTestDataset.yaml

4. A quick export on toyDataset model: 
python train.py -c config/train_toyTestDataset.yaml --export_only


############
To deal with errors like (Mac CPU): "OSError: [Errno 24] Too many open files"
Try to set "ulimit -n 50000" or a bigger number. Then you can use "ulimit -a" to check.

