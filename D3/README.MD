# Reproduction for D³: Scaling Up Deepfake Detection by Learning from Discrepancy

This repository is based on the official implementation of  
**[D³: Scaling Up Deepfake Detection by Learning from Discrepancy (CVPR 2025)](https://arxiv.org/pdf/2404.04584)**.  
👉 Original code is available at [Official GitHub Repo](https://github.com/BigAandSmallq/D3?tab=readme-ov-file).

---

## 📂 Dataset Download

The dataset used in this project is a **composite dataset** provided by the D³ authors:  
**Make sure you have `wget`, `unzip`, and `7z` installed.**

#### 📥 UFD Dataset Download Scripts
The **UFD Dataset** is sourced from the paper ["Towards Universal Fake Image Detectors that Generalize Across Generative Models"](https://arxiv.org/pdf/2302.10174) ([GitHub](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main?tab=readme-ov-file)).  

Part of the data comes from previous work ["Detecting CNN-Generated Images"](https://arxiv.org/pdf/1912.11035) ([GitHub](https://github.com/PeterWang512/CNNDetection/tree/master?tab=readme-ov-file)). First download dataset following the below.

<details>
<summary><strong>Download <code>training set</code></strong></summary>

Create a file named <code>download_trainset.sh</code> with the following content:

```bash
#!/bin/bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
wait $(jobs -p)

7z x progan_train.7z.001
rm progan_train.7z.*
unzip progan_train.zip
rm progan_train.zip
```
</details>

<details>
<summary><strong>Download <code>validation set</code></strong></summary>

Create a file named <code>download_valset.sh</code> with the following content:

```bash
#!/bin/bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip

unzip progan_val.zip
rm progan_val.zip
```
</details>

<details>
<summary><strong>Download <code>test set</code></strong></summary>

Create a file named <code>download_testset.sh</code> with the following content:
```bash
#!/bin/bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip

unzip CNN_synth_testset.zip
rm CNN_synth_testset.zip
```
</details>


<details>
<summary><strong>Re-Order Folders & Files</strong></summary>

Run the below shell file in each folders.

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p 0_real 1_fake

flatten_group() {
  local group="$1"
  find . -regextype posix-extended -type f \
       -regex ".*/${group}/.*" -not -path "./${group}/*" -print0 |
  while IFS= read -r -d '' f; do
    base="${f##*/}"
    dest="./${group}/$base"

    if [[ -e "$dest" ]]; then
      i=1; name="${base%.*}"; ext="${base##*.}"
      [[ "$name" == "$ext" ]] && ext=""
      while [[ -e "$dest" ]]; do
        [[ -n "$ext" ]] && dest="./${group}/${name}__dup${i}.${ext}" || dest="./${group}/${name}__dup${i}"
        ((i++))
      done
    fi

    mv -- "$f" "$dest"
  done
}

flatten_group 0_real
flatten_group 1_fake
echo "✅ Flattened without overwriting."

find . -mindepth 1 -maxdepth 1 -type d ! -name "0_real" ! -name "1_fake" ! -name "run.sh" -exec rm -rf {} +
echo "🧹 Cleaned up all other folders."
```
</details>

<table>
<tr>
  <th rowspan="2">Split</th>
  <th colspan="3">ProGAN</th>
  <th colspan="1">BigGAN</th>
  <th colspan="1">CRN</th>
  <th colspan="1">CycleGAN</th>
  <th colspan="1">DeepFake</th>
  <th colspan="1">GauGAN</th>
  <th colspan="1">IMLE</th>
  <th colspan="1">SAN</th>
  <th colspan="1">SeeingDark (SITD)</th>
  <th colspan="1">StarGAN</th>
  <th colspan="1">StyleGAN</th>
  <th colspan="1">StyleGAN2</th>
  <th colspan="1">whichfaceisreal</th>
</tr>
<tr>
  <th>Train</th>
  <th>Val</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
  <th>Test</th>
</tr>
<tr>
  <th># of Real Images</th>
  <td><center>360,059</center></td>
  <td><center>4,000</center></td>
  <td><center>4,000</center></td>
  <td><center>2,000</center></td>
  <td><center>6,382</center></td>
  <td><center>1,321</center></td>
  <td><center>2,707</center></td>
  <td><center>5,000</center></td>
  <td><center>6,382</center></td>
  <td><center>219</center></td>
  <td><center>180</center></td>
  <td><center>1,999</center></td>
  <td><center>5,991</center></td>
  <td><center>7,988</center></td>
  <td><center>1,000</center></td>
</tr>
<tr>
  <th># of Fake Images</th>
  <td><center>360.060</center></td>
  <td><center>4,000</center></td>
  <td><center>4,000</center></td>
  <td><center>2,000</center></td>
  <td><center>6,382</center></td>
  <td><center>1,321</center></td>
  <td><center>2,698</center></td>
  <td><center>5,000</center></td>
  <td><center>6,382</center></td>
  <td><center>219</center></td>
  <td><center>180</center></td>
  <td><center>1,999</center></td>
  <td><center>5,991</center></td>
  <td><center>7,988</center></td>
  <td><center>1,000</center></td>
<tr>
  <th rowspan="1">Total</th>
  <th colspan="1">720,119</th>
  <th colspan="1">8,000</th>
  <th colspan="1">8,000</th>
  <th colspan="1">4,000</th>
  <th colspan="1">12,764</th>
  <th colspan="1">2,642</th>
  <th colspan="1">5,405</th>
  <th colspan="1">10,000</th>
  <th colspan="1">12,764</th>
  <th colspan="1">438</th>
  <th colspan="1">360</th>
  <th colspan="1">3,998</th>
  <th colspan="1">11,982</th>
  <th colspan="1">15,976</th>
  <th colspan="1">2,000</th>
</tr>
</table>

Another part of the dataset (diffusion models) can be downloaded from the following [Google Drive link](https://drive.usercontent.google.com/download?id=1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t&export=download&authuser=0)

# Note that these images are not leveraged mostly. (Only done in some papers)

<table>
<tr>
  <th rowspan="2">Split</th>
  <th colspan="1">DALL·E</th>
  <th colspan="1">ImageNet</th>
  <th colspan="1">GLIDE(100_10)</th>
  <th colspan="1">GLIDE(100_27)</th>
  <th colspan="1">GLIDE(50_27)</th>
  <th colspan="1">Guided</th>
  <th colspan="1">LAION</th>
  <th colspan="1">LDM(100)</th>
  <th colspan="1">LDM(200)</th>
  <th colspan="1">LDM(200_CFG)</th>
</tr>
<tr>
  <th colspan="10">Test</th>
</tr>
<tr>
  <th># of Real Images</th>
  <td><center>-</center></td>
  <td><center>1,000</center></td>
  <td><center>-</center></td>
  <td><center>-</center></td>
  <td><center>-</center></td>
  <td><center>-</center></td>
  <td><center>1,000</center></td>
  <td><center>-</center></td>
  <td><center>-</center></td>
  <td><center>-</center></td>
</tr>
<tr>
  <th># of Fake Images</th>
  <td><center>1,000</center></td>
  <td><center>-</center></td>
  <td><center>1,000</center></td>
  <td><center>1,000</center></td>
  <td><center>1,000</center></td>
  <td><center>1,000</center></td>
  <td><center>-</center></td>
  <td><center>1,000</center></td>
  <td><center>1,000</center></td>
  <td><center>1,000</center></td>
<tr>
  <th rowspan="1">Total</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
  <th colspan="1">1,000</th>
</tr>
</table>


#### 📥 GenImage Dataset Download Scripts
The **GenImage Dataset** is sourced from the paper ["GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image"](https://arxiv.org/pdf/2306.08571) ([GitHub](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main?tab=readme-ov-file)).  

The datasets can be downloaded from the following [Google Drive link](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS).

<table>
<tr>
  <th rowspan="2">Split</th>
  <th colspan="2">ADM</th>
  <th colspan="2">BigGAN</th>
  <th colspan="2">GLIDE</th>
  <th colspan="2">Midjourney</th>
  <th colspan="2">SDv4</th>
  <th colspan="2">VQDM</th>
  <th colspan="2">wukong</th>
</tr>
<tr>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
  <th colspan="1">Train</th>
  <th colspan="1">Val</th>
</tr>
<tr>
  <th># of Real Images</th>
  <td><center>157,453</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>161,701</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>160,739</center></td>
  <td><center>6,000</center></td>
</tr>
<tr>
  <th># of Fake Images</th>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
  <td><center>162,000</center></td>
  <td><center>6,000</center></td>
<tr>
  <th rowspan="1">Total</th>
  <th colspan="1">319,453</th>
  <th colspan="1">12,000</th>
  <th colspan="1">324,000</th>
  <th colspan="1">12,000</th>
  <th colspan="1">324,000</th>
  <th colspan="1">12,000</th>
  <th colspan="1">323,701</th>
  <th colspan="1">12,000</th>
  <th colspan="1">324,000</th>
  <th colspan="1">12,000</th>
  <th colspan="1">324,000</th>
  <th colspan="1">12,000</th>
  <th colspan="1">322,739</th>
  <th colspan="1">12,000</th>
</tr>
</table>

##### Originally, there is SDv5 in the GenImage Dataset, however, following D³ we only leverage SDv4.

After downloading, extract the files and place them in the appropriate directory as required by your training or evaluation scripts.

---

## ✅ D³ Reproduction (Evaluation Only)

Before doing anything else, first download the required packages by running the command: `pip install -r requirements.txt`.

For evaluation, you can use `validate_for_robustness.py` directly, or run the provided `eval.sh` shell script. You may also execute the evaluation commands one by one in the command line.

In `eval.sh`, the required arguments include `--dataset` and `--gen_model`:
- `--dataset`: options are `GenImage` or `UFD`
- `--gen_model`: e.g., `stylegan`, `cyclegan`, `adm`, etc.

<table>
<tr>
  <th rowspan="2">Method</th>
  <th colspan="8">In-Domain</th>
  <th colspan="12">Out-of-Domain</th>
</tr>
<tr>
  <th>ADM</th>
  <th>BigGAN</th>
  <th>GLIDE</th>
  <th>Midjourney</th>
  <th>LDM</th>
  <th>VQDM</th>
  <th>wukong</th>
  <th>ProGAN</th>
  <th>CycleGAN</th>
  <th>StyleGAN</th>
  <th>StyleGAN2</th>
  <th>GauGAN</th>
  <th>StarGAN</th>
  <th>Deepfakes</th>
  <th>whichfaceisreal</th>
  <th>SITD</th>
  <th>SAN</th>
  <th>CRN</th>
  <th>IMLE</th>
  <th>DALL·E</th>
</tr>
<tr>
  <td><strong>Reported</strong></td>
  <td><center>94.8</center></td>
  <td><center>98.5</center></td>
  <td><center>95.0</center></td>
  <td><center>96.8</center></td>
  <td><center>94.4</center></td>
  <td><center>96.7</center></td>
  <td><center>97.1</center></td>
  <td><center>99.4</center></td>
  <td><center>92.7</center></td>
  <td><center>94.9</center></td>
  <td><center>95.7</center></td>
  <td><center>98.1</center></td>
  <td><center>96.0</center></td>
  <td><center>67.7</center></td>
  <td><center>83.1</center></td>
  <td><center>73.8</center></td>
  <td><center>62.6</center></td>
  <td><center>88.1</center></td>
  <td><center>95.0</center></td>
  <td><center>92.8</center></td>
</tr>
<tr>
  <th>Evaluated</th>
  <td><center>96.6</center></td>
  <td><center>97.9</center></td>
  <td><center>96.8</center></td>
  <td><center>96.7</center></td>
  <td><center>97.3</center></td>
  <td><center>96.6</center></td>
  <td><center>97.0</center></td>
  <td><center>99.3</center></td>
  <td><center>92.1</center></td>
  <td><center>95.1</center></td>
  <td><center>95.6</center></td>
  <td><center>98.0</center></td>
  <td><center>95.8</center></td>
  <td><center>68.2</center></td>
  <td><center>82.8</center></td>
  <td><center>74.7</center></td>
  <td><center>60.3</center></td>
  <td><center>87.9</center></td>
  <td><center>94.8</center></td>
  <td><center>X</center></td>
</table>

"Reported" refers to the results presented in Table 3 of the D³ paper, while "Evaluated" indicates the results obtained by running evaluation using the checkpoint provided by D³.

##### There are slight performance differences due to discrepancies between the number of images reported in the paper and the actual number of images available after the download.
