# PERCH-CXR

<p>
This is a tool to detect World Health Organization (WHO)-defined chest radiograph <a href="https://apps.who.int/iris/bitstream/handle/10665/66956/WHO_V_and_B_01.35.pdf;jsessionid=BBBC54AAF1AC3A4330B6B0C39914412A?sequence=1">primary-endpoint pneumonia</a>, written in TensorFlow 2.2.
</p>
<p>
The project is built on a <a href="https://arxiv.org/abs/1608.06993">DenseNet121</a>, trained using CXR images from the <a href="https://academic.oup.com/cid/article/64/suppl_3/S253/3858215"> Pneumonia Etiology Research for Child Health (PERCH)</a> study. 
The project also used <a href="https://stanfordmlgroup.github.io/competitions/chexpert/">CheXpert</a> dataset for pretraining, and two WHO datasets: <a href="https://pubmed.ncbi.nlm.nih.gov/15976876/">WHO-original</a> and <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5608771/">WHO-CRES (Chest Radiography in Epidemiological Studies)</a> for testing.
</p>
![alt text](WHO_images/OI.png)

![Tux, the Linux mascot](/assets/images/tux.png)

  
