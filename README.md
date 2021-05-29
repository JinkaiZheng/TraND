# TraND
This is the code for the paper "Jinkai Zheng, Xinchen Liu, Chenggang Yan, Jiyong Zhang, Wu Liu, Xiaoping Zhang and Tao Mei: [TraND: Transferable Neighborhood Discovery for
Unsupervised Cross-domain Gait Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9401218). ISCAS 2021" (**Best Paper Award - Honorable Mention**)


## Requirements
- Conda
- GPUs
- Python 3.7
- PyTorch 1.1.0

### Installation
You can replace the second command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/#v110) 
based on your CUDA version.
```
git clone https://github.com/JinkaiZheng/TraND.git
cd TraND
conda create --name py37torch110 python=3.7
conda activate py37torch110
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements
```


## Data Preparation
Download [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) and [OU-LP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLP.html)

### Data Pretreatment
`pretreatment_casia.py` and `pretreatment_oulp.py` use the alignment method in
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
In the case of CASIA-B dataset, you need to run the command:
```
python GaitSet/pretreatment_casia.py --input_path='root_path_of_raw_dataset' --output_path='./data/CASIA-B'
```

### Data Structrue
After the pretreatment, the data structure under the directory should like this
```
./data
├── CASIA-B
│  ├── 001
│     ├── bg-01
│        ├── 000
│           └── 001-bg-01-000-001.png
├── OULP
│  ├── 0000024
│     ├── Seq00
│        ├── 55
            └── 00000061.png
```


## Train
### Stage I: Supervised Prior Knowledge Learning on Source Domain

Training the GaitSet model in the source domain, run this command:
```bash
 python GaitSet/train.py --data "casia-b"
```
### Stage II: Transferable Neighbor Discovery on Target Domain

Fine-tuning the GaitSet model in the target domain with TraND method, run this command:
```bash
sh Experement.sh
```


## Test
Testing the model in self domain, such as CASIA-B dataset, run this command:
```
python GaitSet/test.py --data "casia-b"
```
Testing the model in cross domain, such as CASIA-B -> OU-LP dataset, run this command:
```
python GaitSet/test_cross.py --source "casia-b" --target "oulp"
```


## Citation
Please cite this paper in your publications if it helps your research:
```
@article{DBLP:journals/corr/abs-2102-04621,
  author    = {Jinkai Zheng and
               Xinchen Liu and
               Chenggang Yan and
               Jiyong Zhang and
               Wu Liu and
               Xiaoping Zhang and
               Tao Mei},
  title     = {TraND: Transferable Neighborhood Discovery for Unsupervised Cross-domain
               Gait Recognition},
  journal   = {ISCAS},
  year      = {2021}
}
```


## Acknowledgement
- [GaitSet](https://github.com/AbnerHqC/GaitSet)
- [AND](https://github.com/Raymond-sci/AND)
