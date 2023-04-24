# ECE-GY-7123-Deep-Learning-Project-Spring-2023 (WIP)
Repository for project - working title `LYCB`

### To Do
1. Fix blender script transform matrix / coords output. Seems to be wrong still based on MP4 output; though it still converges.
2. Output mesh is no good - dress hole is filled as expected.
3. Output mesh is textureless!
4. PERGAMO integration

---------
### How to Run
```shell
#Setup
git clone https://github.com/datacrisis/ECE-GY-7123-Deep-Learning-Project-Spring-2023.git
cd "ECE-GY-7123-Deep-Learning-Project-Spring-2023"

pip install -r requirements.txt
```
```python
#Run
from torchngp import torchngp
PSNR,LPIPS,loss,trainer = torchngp(data_path='data/fox',
                          workspace='output/fox_text_output',
                          iters=1500,
                          scale=0.33)
```
