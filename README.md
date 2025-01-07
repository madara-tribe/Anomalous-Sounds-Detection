# Anomalous-Sounds-Detection

This repository is useful Network to detect sound detection or similar task.
It includes 1DCNN and 2DCNN for wave data. 

the method to convert wave feature to image through matplotlib is charastalistic.

### Used task and dataset 

- [DACASE 2024](https://dcase.community/challenge2024/index)

# Performance (Pink noise detection)

Basic performance of each models by Pink noise detection task

### 1DCNN

<img src="https://github.com/user-attachments/assets/97cb9017-642b-4198-9878-02f14c6590c5" width="400px" height="300px"/><img src="https://github.com/user-attachments/assets/234839c4-b046-4f57-bd6c-afe61a5435c1" width="400px" height="300px"/>







### WaveVisonNet

<img src="https://github.com/user-attachments/assets/b0feb2f0-a86d-49ec-82e8-5773b5530a02" width="400px" height="300px"/><img src="https://github.com/user-attachments/assets/88cafd28-e31d-49da-b41e-8e1f09824de1" width="400px" height="300px"/>


### VAE for 1D

<img src="https://github.com/user-attachments/assets/b0feb2f0-a86d-49ec-82e8-5773b5530a02" width="400px" height="300px"/><img src="https://github.com/user-attachments/assets/88cafd28-e31d-49da-b41e-8e1f09824de1" width="400px" height="300px"/>


# convert wave to image through matplotlib

<b>this is the way to convert wave feature to image through matplotlib</b>

![plt_process](https://github.com/user-attachments/assets/3f990b7c-3d18-4ebb-8541-2e0095e81a4f)


```python
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def convert_waveform_to_image(waveform):
    waveform = waveform.squeeze()
    plt.figure(figsize=(10, 4)) 
    plt.imshow(waveform, aspect='auto')
    plt.axis('off') 
    # Save the figure to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    return image
```


# References
- [DCASE-Task1A-Code](https://github.com/WangHelin1997/DCASE-2020-Task1A-Code/tree/update)
- [機械音の異常検知チャレンジ DCASE 2020 Task 2](https://qiita.com/daisukelab/items/b106c567cf8927a5519a)
- [Kaggle G2Net Gravitational Wave Detection コンペ振り返り](https://qiita.com/anonamename/items/5b7fa5d9d5d7f9970e06)
