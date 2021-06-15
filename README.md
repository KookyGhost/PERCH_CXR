<h1 id="perch_cxr">PERCH_CXR</h1>
<p> <a href="https://apps.who.int/iris/bitstream/handle/10665/66956/WHO_V_and_B_01.35.pdf;jsessionid=BBBC54AAF1AC3A4330B6B0C39914412A?sequence=1">Primary endpoint pneumonia (<strong>PEP</strong>)</a> along with non-endpoint/other infiltrates (<strong>OI</strong>), and <strong>non-PEP/OI</strong> are defined by the World Health Organization (WHO) as endpoints of evaluation in vaccine effectiveness studies among pediatric population. This is a tool to detect these endpoints on pediatric chest radiograph. The project is written in <strong> TensorFlow 2.2</strong>. The original study was published on [preprint link to be added].</p>
<p>The model was built using <a href="https://arxiv.org/abs/1608.06993">DenseNet121</a> and trained using chest x-ray images from the <a href="https://academic.oup.com/cid/article/64/suppl_3/S253/3858215"> Pneumonia Etiology Research for Child Health (PERCH)</a> study. 
The project also used <a href="https://stanfordmlgroup.github.io/competitions/chexpert/">CheXpert</a> dataset for pretraining, and images from <a href="https://pubmed.ncbi.nlm.nih.gov/15976876/">WHO-original</a> and <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5608771/">WHO-CRES (Chest Radiography in Epidemiological Studies)</a> for testing.</p>
<p><strong>Disclaimer:</strong> The definition is designed to be used in epidmiological studies which typically favors <strong>specificty</strong> over sensitivity, and thus is not meant to be used for clinical diagnosis where a higher sensitivity is preferred to reduce under-diagnosis. <a href="https://pubmed.ncbi.nlm.nih.gov/21870077/">Some researchers</a> have recommended dropping OI category from the WHO definition due to its low inter-rater agreement.</p>
<h2 id="visualization">Visualization</h2>
<h3 id="primary-endpoint-pneumonia">Primary Endpoint Pneumonia</h3>
<details>
  <summary>Click to expand</summary><br>
<img src="WHO_images/PEP.png" alt="alt text" />
Frontal radiographs of the chest in a child with WHO-defined primary endpoint pneumonia; the child is rotated to the right with dense opacity in the right upper lobe; the model localizes consolidation with a predicted probability p = 0.980; the discriminative visualization shows fine-grained features important to the predicted class.
</details>

<h3 id="non-endpoint-other-infiltrate">Non-Endpoint/Other Infiltrate</h3>
<details>
  <summary>Click to expand</summary>

<img src="WHO_images/OI.png" alt="alt text" />
Frontal radiograph of the chest presents patchy opacity consistent with non-endpoint infiltrate. The model correctly classifies the image as infiltrate with a probability of p = 0.917 and localizes the areas of opacity. The class discriminative visualization highlights important class features.
</details>

<h2 id="user-tutorial">User Tutorial</h2>
<h3 id="file-structure-">File Structure:</h3>
<p>The folder <a href="./WHO_images">WHO_images</a> contains a toy sample of 9 randomly selected images (3 PEP, 3 OI, and 3 non-PEP/OI) from the WHO-CRES dataset.
The folder <a href="./saved_model">saved_model</a> contains a pre-trained weight file from the PERCH-CXR study.</p>
<h3 id="step-by-step-instructions-">Step by Step Instructions:</h3>
<p><strong>Note:</strong> The instructions work for the toy sample right out of the box. For training on your own data, modify the parameters in <code>config.ini</code>, and structure the data csv file as is in <a href="./WHO_images/WHO_CRES.csv">WHO_CRES.csv</a>, with first column containing path to each image, and second onward column containing image labels.</p>
<ol>
<li>Run <code>python generate_tfreocrd.py</code> to transform data into <a href="https://www.tensorflow.org/tutorials/load_data/tfrecord">TFRecords</a> file, an optional format for TensorFlow, recommended for working large dataset.</li>
<li>Run <code>python train.py</code> to train and evaluate the model.</li>
<li>Run <code>python test.py</code> to test the model.</li>
<li>Run <code>python grad-cam.py</code> to visualize model&#39;s prediction using <a href="https://arxiv.org/abs/1610.02391">Grad-CAM</a>.</li>
</ol>
<h3 id="config-ini-file-explanation">Config.ini File Explanation</h3>
<ul>
<li>The file contains 4 sections (<code>DATA</code>,<code>TRAIN</code>,<code>TEST</code>,<code>GRAD-CAM</code>), each corresponding to one of the 4 steps.</li>
<li>The <code>[DATA]</code> section contains a <code>sharding</code> parameter. Sharding can be used to acheive a more thoroughly shuffled dataset in order to destroy any large-scale correlations in your data (see <a href="https://www.moderndescartes.com/essays/shuffle_viz/">&quot;How to shuffle in TensorFlow&quot;</a>).</li>
<li>When <code>n_fold</code>&gt;1, the training will run in n-fold cross-validation mode, and the results will be saved to n folders. </li>
<li>The <code>[TRAIN]</code> section contains a <code>class_names</code> parameter. Its order corresponds to the order of outcome columns in the data csv file and determines the order of model outputs. If you are evaluating your pre-trained weight in a new dataset, remember to sort the outcome columns in your new data according to <code>class_names</code>.</li>
</ul>
<h2 id="author">Author</h2>
<p>Star Chen (starchen1440@gmail.com)</p>
<h2 id="acknowledgment">Acknowledgment</h2>
<p>The study is sponsored by <a href="https://www.merck.com/">Merck &amp; Co., Inc.</a></p>
<h2 id="license">License</h2>
<p>This project is licensed under the terms of the MIT license.</p>
