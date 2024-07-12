  <h2>🦄️ Noise Calibration: Plug-and-play Content-Preserving Video Enhancement using Pre-trained Video Diffusion Models (ECCV 2024) </h2>

<div>
    <a href='https://github.com/yangqy1110' target='_blank'>Qinyu Yang</a> <sup>1</sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=6UPJSvwAAAAJ&hl=zh-CN' target='_blank'>Haoxin Chen</a><sup>2</sup> &nbsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang</a><sup>2,*</sup> &nbsp; 
    <a href='https://menghanxia.github.io/' target='_blank'>Menghan Xia</a><sup>2</sup> &nbsp; 
    <a href='https://vinthony.github.io/academic/' target='_blank'>Xiaodong Cun</a><sup>2</sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=ycFs33AAAAAJ&hl=en' target='_blank'>Zhixun Su</a><sup>1,*</sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en' target='_blank'>Ying Shan</a><sup>2</sup> &nbsp;
</div>
<div>
    <sup>1</sup> Dalian University of Technology &nbsp; <sup>2</sup> Tencent AI Lab &nbsp; <sup>*</sup> Corresponding Author &nbsp; 
</div>

In *European Conference on Computer Vision (ECCV) 2024*

<a href=''><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://yangqy1110.github.io/NC-SDEdit/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

## Introduction
We propose Noise Calibration,a method that substantially improves consistency of content between enhanced videos based on SDEdit and original videos.

✅ Totally <span style="color: red; font-weight: bold">no</span> training &nbsp;&nbsp;&nbsp;&nbsp;
✅ Less than <span style="color: red; font-weight: bold">10%</span> extra time &nbsp;&nbsp;&nbsp;&nbsp;
✅ Plug-and-play <span style="color: red; font-weight: bold"></span>  &nbsp;&nbsp;&nbsp;&nbsp;


<td><a href="https://github.com/yangqy1110/NC-SDEdit/tree/main/docs/video.gif"><img src=docs/video.gif width="1000"></td>


## Code of Noise Calibration

```Python
import torch
import torch.fft as fft


def get_low_or_high_fft(x, scale, is_low=True):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, T, H, W = x_freq.shape
    
    # extract
    if is_low:
        mask = torch.zeros((B, C, T, H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 1
    else:
        mask = torch.ones((B, C, T, H, W), device=x.device)
        crow, ccol = H // 2, W //2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 0
    x_freq = x_freq * mask
    
    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtere
-------------------------------------------------------------------------------------------------------------------------------------------------------
### Noise Calibration(Algorithm 1)
e_t = torch.randn_like(x_r)
for _ in range(N):
  x = a_t.sqrt() * x_r + sqrt_one_minus_at * e_t
  e_t_theta = self.model.apply_model(x, t, c, **kwargs) # 
  x_0_t = (x - sqrt_one_minus_at * e_t_theta) / a_t.sqrt()
  e_t = e_t_theta + a_t.sqrt() / sqrt_one_minus_at * (get_low_or_high_fft(x_0_t, scale, is_low=False) - get_low_or_high_fft(x_r, scale, is_low=False))

x = a_t.sqrt() * x_r + sqrt_one_minus_at * e_t
------------------------------------------------------------------------------------------------------------------------------------------------------
Next, you can use SDEdit to enhance the video based on x.
```
