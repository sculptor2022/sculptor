# SCULPTOR: Skeleton-Consistent Face Creation Using a Learned Parametric Generator

We present SCULPTOR, a skeleton-consistent face generator that jointly models the skull, face geometry, and face appearance, allowing for high-quality and characteristic facial detail generation.  

## Installation

SCULPTOR is built on **Python 3.7**.  

Setup the SCULPTOR `conda` environment.

```
$ conda create --name SCULPTOR python==3.7
$ conda activate SCULPTOR
```

Clone the repository.  

```
$ git clone https://github.com/sculptor2022/sculptor.git
```


Install the requirements in `requirements.txt`.

```
$ pip install -r requirements.txt
```

## Model
SCULPTOR model is saved in `model/paradict.npy`, which can be downloaded [here](https://github.com/sculptor2022/sculptor/blob/main/model/paradict.npy). We currently provide 50 shape components and 10 trait components publicly available. The forward layer is provided.  

## LUCY dataset

We actively collaborate with orthognathic surgeons to collect real-world data that shows the skeleton-consistent variation of the facial outer surface.   

Orthognathic surgery is a plastic surgery that optimizes facial proportions to treat functional problems caused by bite discrepancies or facial imbalances.

We present LUCY, a comprehensive shape-skeleton correlated face dataset from pre- and post-surgery CT imaging and 3D scans. LUCY consists of 144 scans of anonymized 72 subjects. The 3D maxillofacial CT imaging was performed using a spiral CT scanner (Light speed 16; GE, Gloucestershire, UK), with image spatial resolution $0.48 \times 0.48 \times 1~mm^3$.

We provide one subject pre- and post-surgery CT imaging sample in this repository, `LUCY_sample`.  

### Requesting LUCY

A handwritten agreement must be signed by both the recipient and the research administration office director of your institution to obtain more data in LUCY. Students are not qualified to request Please have your supervisor submit the form.  

Send the handwritten agreement form to sculptor2022@outlook.com. We will carefully verify each agreement form. 

LUCY is available for non-commercial research and education purposes only. LUCY must not be reproduced, exchanged, sold, or used for profit. All publishable work using any of the LUCY or SCULPTOR,  following paper must be cited.

```
@article{

}
```






## Acknowledgement 

This repository is built with modifications from [SMPLX](https://github.com/vchoutas/smplx) and [FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch.git).