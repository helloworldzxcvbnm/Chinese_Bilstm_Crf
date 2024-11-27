# 基于 Bilstm Crf 的中文命名实体识别

## 依赖
- python3.7
- pytorch 1.10.0 
- CUDA 11.6
- sklearn
- torchcrf
- GPU RTX3060 一张

## 项目结构
Chinese_Bilstm_Crf </br>
├── down_data.py     （下载transformer内置数据集，需要挂梯子） </br>
├── id_to_label_msra_ner.txt     （数字对应BIO的标签字典）</br>
├── README.md </br>
├── run.ipynb （运行程序，jupyter版本，推荐）</br>
├── run.py （运行程序，py版本）</br>
├── show_GPU_Utilization.py （实时显示GPU使用率）</br>
├── tags.txt （数据文件———前10000条标签）</br>
├── tags_total.txt （数据文件———全部标签）</br>
├── texts.txt （数据文件———前10000条文本）</br>
├── texts_total.txt （数据文件———全部文本）</br>
### 1. 该项目只实现了基本功能，包括输出统计训练集和测试集对应的三个命名实体的数量，每次迭代loss以及训练完成后相应的评估指标。
### 2. 推荐使用ipynp版本，训练完成后，可以自行输入单句查看识别效果。
### 3. 数据集采用的是 msra_ner 的前10000条数据。如果想使用完整（45000条）数据，请将run里面载入数据的文件名替换为完整数据的文件名。
### 4. Batch_Size 请根据自己电脑性能设置（一般为40或32），如果不想让电脑变成直升飞机可以使用16，但训练速度也会缓慢。
### 5. Epoch 推荐 20（10或15也是可以的，但效果相对较差）。
### （修改准确率计算方式，排除O标签，其他没变）
