# TextClassify
HFUT Soft Engineering Competition.
本项目已上传至[Github](https://github.com/ZonePG/TextClassify)

## Requirements
安装项目所需依赖
```
pip3 install -r requirements.txt
```

## 对数据集进行预测测试
`cd market` 进入 market 文件夹运行`python3 test.py`，输入待测试的xlsx文件，输出为填充好的`output.xlsx`，均设置在market路径下
这里，我们已经将测试文件`test_data.xlsx`放入了market文件夹，
输出文件`output.xlsx`输出在market文件夹中，可以直接查看结果

## 部署 Web 网站
`cd` 到项目主路径
### On Windows
```
set FLASK_APP=run.py
set FLASK_DENUG=True
flask run
```
### On Mac/Linux 
```
export FLASK_APP=run.py
export FLASK_DENUG=True
flask run
```
在网页中输入`http://127.0.0.1:5000/`即可查看在线网页。

## Reference and Thanks
- [EasyBert](https://github.com/rsanshierli/EasyBert)
- [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
- [Flask Course](https://www.youtube.com/watch?v=Qr4QMBUPxWo)
