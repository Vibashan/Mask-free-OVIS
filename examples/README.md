## Illustration of Pseudo-mask Generation

### Object Proposals

* Download our proposals [here](https://drive.google.com/drive/folders/13-4Ct538mV_tCkK0mADHOviYr0-vBdf0?usp=sharing) and put under this folder

    ```
    ./proposals/
    ├── coco/
    |   ├── images/
    │       ├── train2017/*.pkl
    ```

* each .pkl file contains a list of numpy.ndarray [n_1 * 5, n_2 * 5,...,n_m * 5]

* the i_th numpy.ndarray correspond to n_i proposals in [xmin, ymin, xmax, ymax, score] obtained from WSPN.
* each _info.pkl contains image information

### Image Caption Dataset

* We provide an example of image-caption dataset in image_caption_final.json

### Object of Interest

* A train vocabulary contains objects of interest is in object_vocab.json

### Download ALBEF Pre-trained Model

* Download ALBEF pre-trained checkpoint [ALBEF.pth](https://github.com/salesforce/ALBEF#download) and put under this folder
