# OpenCVFaceRecog

## faces.py
Will used the learned data (in pickels and recognizers) to idenfiy a person or object in live video feed.

## faces_train.py
Will help you in training the model with static images places inside images folder. (refer readme inside it)

## faces_train_live.py
Will help you in training the model with live video feed. This can be used to train multiple people within same live feed but only one person at a time.

## better_face_recog_train.py
Will help in training the faces that the system will recognise. As of not it works for single face only. It is an improved version of faces_train_live.py in the sense that it can detect side faces as well rotated to 30-40 deg (approx no fix value). It has better continous detection.

## better_face_recog.py
Will used the learned data (in pickels and recognizers) to idenfiy a person or object in live video feed. It is improved version of faces.py as it is capable of detecting from little side faces as well also with partial covered faces.

For more detials please refer the code as more documentation is not yet done.

> Note: Whole of this code is not developed by me alone. I have taken numerous references from internet to come up with this code. Feel free to use this code and no need to provide a credit as well. :) Happy learning.

> Note 2: I do not remember from where i have taken references and hence as of now its not possible for me to put the references. Please in case if I have used your work or code and you want you reference to be mentioned here, do let me know. This is for learning only and not a meant for commercial use by me.

> Note 3: In case if you have any patch that you would like to be merged, please do let me know with pull request. I'll review it and merge it. Otherwise, I'll create a folder by your name/id and put the merged file there for everyone to see.

## Credits
- https://www.pyimagesearch.com/
- https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/
- https://www.linkedin.com/in/adrian-rosebrock-59b8732a/
