# openposeST-GCN
Inference Gloss from Sign Language  Video with OpenPose and ST-GCN!  
OpenPose와 WLASL Preatrained ST-GCN을 이용한 수화 번역 
  
## 🚀 Import pyopenpose
If you want use pyopenpose, see [here](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_03_python_api.html)  
한국어 가이드를 원하신다면, 제 [블로그](https://blog.naver.com/paragonyun/223160841596)를 방문하셔서 따라하셔도 됩니다.
  
## 👀 How to Use
0. You have to be ready with `pyopenpose` 
1. Clone my repo
```
git clone https://github.com/Probono-sign-language-detection/openposeST-GCN.git
```
2. install what you need
```
pip install -r requirements.txt
```
3. main.py
```
python stgcn_test.py
```

## 🤖 Results
![image](https://github.com/Probono-sign-language-detection/openposeST-GCN/assets/83996346/de1fab2d-783c-4f8e-ae64-0055eb8f3e98)  
You can get label index for your video's frames! But actually it is too slow for CPU Environment.   
So If you want to get a label in real-time task, I recommend to use the setting(import pyopenpose step!) for CUDA Environment.  
