The project contains the 4 major steps, and all parameters you need for training 
can be specified in config.ini. 

Step 1:Generate dataset using generate_tfrecord.py 
Comment:The generate_tfrecord.py is used to convert data into TFRecord format.
	This format is the recommended data format for TensorFlow.
 
	 
Step 2: Train data using train.py
Comment:Use train.py to train the data generated in Step 1.

Step 3: Test data using test.py (TBC)
Comment:Use test.py to test the trained model in Step 2 on new dataset. 
        Similar to training, first convert your new data into TFRecord format,
	using generate_tfrecord.py, and then run test.py.  

Step 4: Generate heat map on CXR using grad-cam.py
Comment: use grad-cam.py to generate heat map on CXR images. This can be used
	to visualize "disease area", or area related to model's prediction on 
	CXR.

