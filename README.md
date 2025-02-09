# Person Image Synthesis via Denoising Diffusion Model [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb)

 <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2211.12500">ArXiv</a>
    | 
    <a href="">Paper</a>
    | 
    <a href="https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb">Demo</a>
  </b>
</p> 
<p align="center">
<img src=Figures/images.gif>

# Getting Started

For Newton, make sure you have Python 3.8 installed:

```bash
module load python/python-3.8.0-gcc-9.1.0
```

Set up and activate virtual environment

```bash
python3 -m venv ./venv
source venv/bin/activate
```

Upgrade pip and install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown tensorfn
```

Run the main script like so:

```bash
python3 src/pidm.py
```

-   NOTE: Make you have a cuda environment setup in order to run this.

# News

-   **2023.02** A demo available through Google Colab:

    :rocket:
    [Demo on Colab](https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb)

-   **2023.02** Codes will be available soon!

# Generated Results

<img src=Figures/itw.jpg>

You can directly download our test results from Google Drive: (1) [PIDM.zip](https://drive.google.com/file/d/1zcyTF37UrOmUqtRwwq1kgkyxnNX3oaQN/view?usp=share_link) (2) [PIDM_vs_Others.zip](https://drive.google.com/file/d/1iu75RVQBjR-TbB4ZQUns1oalzYZdNqGS/view?usp=share_link)

The [PIDM_vs_Others.zip](https://drive.google.com/file/d/1iu75RVQBjR-TbB4ZQUns1oalzYZdNqGS/view?usp=share_link) file compares our method with several state-of-the-art methods e.g. ADGAN [14], PISE [24], GFLA [20], DPTN [25], CASD [29],
NTED [19]. Each row contains target_pose, source_image, ground_truth, ADGAN, PISE, GFLA, DPTN, CASD, NTED, and PIDM (ours) respectively.
Some of the results are shown below.

<p align="center">
<img src=Figures/github_qual.jpg>
</p>

# Citation

If you use the results and code for your research, please cite our paper:

```

@article{bhunia2022pidm,
title={Person Image Synthesis via Denoising Diffusion Model},
author={Bhunia, Ankan Kumar and Khan, Salman and Cholakkal, Hisham and Anwer, Rao Muhammad and Laaksonen, Jorma and Shah, Mubarak and Khan, Fahad Shahbaz},
journal={arXiv},
year={2022}
}

```

[Ankan Kumar Bhunia](https://scholar.google.com/citations?user=2leAc3AAAAAJ&hl=en),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en),
[Rao Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en),
[Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en),
[Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)

```

```
