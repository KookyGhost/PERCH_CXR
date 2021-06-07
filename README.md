# PERCH-CXR


This is a tool to detect <a href="https://apps.who.int/iris/bitstream/handle/10665/66956/WHO_V_and_B_01.35.pdf;jsessionid=BBBC54AAF1AC3A4330B6B0C39914412A?sequence=1">World Health Organization (WHO)-defined chest radiograph primary-endpoint pneumonia (<strong>PEP</strong>)</a>, written in <strong> TensorFlow 2.2</strong>.

The project is built on a <a href="https://arxiv.org/abs/1608.06993">DenseNet121</a>, trained using CXR images from the <a href="https://academic.oup.com/cid/article/64/suppl_3/S253/3858215"> Pneumonia Etiology Research for Child Health (PERCH)</a> study. 
The project also used <a href="https://stanfordmlgroup.github.io/competitions/chexpert/">CheXpert</a> dataset for pretraining, and two WHO datasets: <a href="https://pubmed.ncbi.nlm.nih.gov/15976876/">WHO-original</a> and <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5608771/">WHO-CRES (Chest Radiography in Epidemiological Studies)</a> for testing.

<strong>Disclaimer:</strong> The WHO definition is designed to be used in large epidmiological studies which favors <strong><em>specificty</em></strong> over sensitivity, and is not meant to be used for clinical diagnosis where higher sensitivity is preferred to reduce under-diagnosis. The definition contains 3 outcomes: <strong>PEP</strong>, non-endpoint/other infiltrates (<strong>OI</strong>), and <strong>non-PEP/OI</strong>. The PEP is the main outcome in vaccine studies, and has the highest inter-rater agreement and best model prediction accuracy (AUC = 97.7%) in WHO datasets, and OI has the lowest agreement and accuracy. <a href="https://pubmed.ncbi.nlm.nih.gov/21870077/">Researchers have recommended dropping OI category from the WHO definition</a>.


## Visualization
 ### Primary Endpoint Pneumonia
<details>
  <summary>Click to expand</summary>  
  
![alt text](WHO_images/PEP.png)
Frontal radiographs of the chest in a child with WHO-defined primary endpoint pneumonia; the child is rotated to the right with dense opacity in the right upper lobe; the model localizes consolidation with a predicted probability p = 0.980; the discriminative visualization shows fine-grained features important to the predicted class.
</details>

### Non-Endpoint Infiltrate

<details>
  <summary>Click to expand</summary>

<img src="WHO_images/OI.png" alt="alt text" />
Frontal radiograph of the chest presents patchy opacity consistent with non-endpoint infiltrate. The model correctly classifies the image as infiltrate with a probability of p = 0.917 and localizes the areas of opacity. The class discriminative visualization highlights important class features.
</details>

## User Tutorial
### File Structure:
The folder [WHO_images](./WHO_images) contains a toy sample of 9 randomly selected images (3 PEP, 3 OI, and 3 non-PEP/OI) from the WHO-CRES dataset.
The folder [saved_model[(./saved_model) contains a pre-trained weight file from the PERCH-CXR study.

### Step by Step Instruction:
<strong>Note that this instruction works for the toy sample right out of the box. For training on your own data, modify the parameters in ```config.ini```

1. Run ```python generate_tfreocrd.py``` to transform data into <a href="https://www.tensorflow.org/tutorials/load_data/tfrecord">TFRecords</a> file, the recommended format for TensorFlow.
2. Run ```python train.py``` to train and evaluate the model.
3. Run ```python test.py``` to test the model.
4. Run ```python grad-cam.py``` to visualize model's prediction using <a href="https://arxiv.org/abs/1610.02391">Grad-CAM</a>.



  
