# RecoOnEdge
This is the code repository for Two Watts is All You Need: Enabling In-Detector Real-Time Machine Learning for Neutrino Telescopes Via Edge Computing

### Plots
Currently, you can find all the plots and the scripts to make them (as well as the data) in directory paper_plots.

### Model architecture and hyperparameters
you can find the network architecture, training script, and (hyper-)parameter settings in the directory **./v2_smallnet**.

### Simulation and data
You can find the IceHex detector simulation data (as described in manuscript) in google drive [here](https://drive.google.com/file/d/1VBEYPS3nOc5IW57No7vBVfN7jvuBmu0H/view?usp=share_link). The parquet file is all the Trigger level prometheus simulated IceHex detector data as described in the manuscript; whereas the hdf5 files, ready to be input into the network for training and evaluation using the methods as defined in **./v2_smallnet/H5data.py**, can be obtained by preprocessing these data, or upon reasonable request. 

### Saved keras checkpoints and tflite models
You can find the keras checkpoints for zenith and azimuth prediction for water and ice detectors respectively in **./v2_smallnet/expts/water_final** and **./v2_smallnet/expts/ice_final**, respectively. The tflite models are initialized and produced from the checkpoints using the *initialize_interpreter()* method defined in **./v2_smallnet/utils.py**, if you would like, feel free to save the tflite model in addition to returning it. For compilation into edge tpu comtaible format, the google tpu compiler is employed directly on the tflite models, you can find a detailed description here in [Coral AI documentation](https://coral.ai/docs/edgetpu/compiler/#usage)

### Other information
More detailed documentation is on the way. For now, please contact Miaochen Jin (email found in manuscript) for any question. We are willing to discuss and share any data upon requests.
