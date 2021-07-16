# OCR Baseline

## 数据准备

报名参赛后，训练集和测试集是六个csv文件，文件内部是json格式的字符串，内含每张图片的下载链接以及标注好的Label。

### 下载baseline

https://gitee.com/coggle/tianchi-intel-PaddleOCR

### 步骤1：下载比赛图片
```sh
python3 down_image.py
```
下载照片可能需要很久时间

### 步骤2：下载预测模型

由于OCR包括多个步骤，此时我们只对其中检测的部署进行fientune，所以其他部署的权重也需要下载。

```sh
mkdir inference && cd inference/

# 下载模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar

# 解压模型
tar -xf ch_ppocr_server_v2.0_rec_infer.tar 
tar -xf ch_ppocr_server_v2.0_det_infer.tar
tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar
```

测试

```sh
python3 tools/infer/predict_system.py --image_dir="./1.jpg" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True

```

### 步骤3：训练预检测模型

首先下载检测模块的预训练模型：

```sh
cd inference
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
tar -xf ch_ppocr_server_v2.0_det_train.tar
```

然后进行finetune，这里训练4个epoch，30分钟左右完成训练。

```sh
python3 tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/

```

### 步骤4：对测试集进行预测

训练完成后，接下来需要将模型权重导出，用于预测。并对测试集的图片进行预测，写入json。

```sh
# 将模型导出
python3 tools/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model=output/ch_db_res18/best_accuracy  Global.save_inference_dir=output/ch_db_res18/

# 对测试集进行预测
python3 tools/infer/predict_system_tianchi.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="output/ch_db_res18/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True

# 将结果文件压缩
zip -r submit.zip Xeon1OCR_round1_test*
```

## 提交说明

要求参与者使用一个json文件提交所有图像的预测结果，结果格式如下：
```json
{

“ res_1”：[

{“points”：[[x 1，y 1 ]，[x 2，y 2 ]，…，[x n，y n ]]，“confidence”：c，“transcription”：“ trans1”}，

…

{“points”：[[x 1，y 1 ]，[x 2，y 2 ]，…，[x n，y n ]]，“confidence”：c，“transcription”：“ trans2”}]，

“ res_2”：[

{“points”：[[x 1，y 1 ]，[x 2，y 2 ]，…，[x n，y n ]]，“confidence”：c，“transcription”：“ trans3”}]，

……

}
```
提交需注意：本赛道对外提供的测试集name_test.csv下，用户的结果文件对应为name_test.json。选手上传的文件需要和真实标注命名一致。

## 评估标准

评测指标时，我们首先通过计算检测区域与相应的真实标注的交并比（IoU）来评估检测区域。IoU值高于0.5的检测区域将被认为成功匹配真实的标注框（即特定文本区域的真实标注）。同时，在有多个匹配项的情况下，我们仅考虑具有最高IOU值的检测区域，其余匹配项将被视为误检。最终评测结果只用1-N.E.D作为正式排名的指标。

然后，我们将使用准确度1-N.E.D来评估模型的预测识别能力。我们会采用归一化编辑距离度量（特别是1-N.E.D）和单词精度来作为模型效果的参考。注意：我们会发布两个指标的结果，但只把1-N.E.D作为正式排名的指标(1-N.E.D_recall, 1-N.E.D_precision, 1-N.E.D_hmean)。

本次比赛的参考指标，和icdar的评测逻辑基本保持一致。

1-N.E.D****度量指标（Normalized Edit Distance metric）：

归一化编辑距离（1-N.E.D）的公式如下：

$$
Norm=1-\frac{1}{N} \sum_{i=1}^{N} \frac{D(s_{i},\hat{s_{i}}) }{\max (s_{i},\hat{s_{i}})}
$$

其中D代表的Levenshtein Distance，也是识别字符的编辑距离，和 $s_{i}$与 $\hat{s_{i}}$ 表示字符串中的预测文本和对应的真实值的区域。注意，对应真实值的区域 挑选的规则是，计算所有真实值和预测值 的IOU，将最大的IOU值对应的真实值和预测值作为匹配对进行计算。N是"匹配对"（ 真实值和预测区域匹配上）的最大数量，其中也包括单例：与任何检测区域都不匹配的真实值区域和与任何真实值区域都不匹配的检测区域。

注意：为了避免注释中的歧义，我们在评估之前会执行某些预处理步骤：

1）繁体字和简体字被视为同一标签；

2）所有的符号都将被识别半角符号；

3）所有难以辨认的图像均不会对评估结果有所帮助。