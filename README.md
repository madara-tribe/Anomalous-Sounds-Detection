# Anomalous-Sounds-Detection



# Performance (Pink noise detection)


## 1DCNN

<img src="https://github.com/user-attachments/assets/97cb9017-642b-4198-9878-02f14c6590c5" width="400px" height="300px"/><img src="https://github.com/user-attachments/assets/234839c4-b046-4f57-bd6c-afe61a5435c1" width="400px" height="300px"/>







## WaveVisonNet

<img src="https://github.com/user-attachments/assets/b0feb2f0-a86d-49ec-82e8-5773b5530a02" width="400px" height="300px"/><img src="https://github.com/user-attachments/assets/88cafd28-e31d-49da-b41e-8e1f09824de1" width="400px" height="300px"/>



# convert wave to image through matplotlib

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

