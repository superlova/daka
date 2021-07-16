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