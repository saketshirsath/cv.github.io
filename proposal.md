### Saket Shirsath, Parth Thakkar, Ben Wolfson
<br>
[Project Update 1](index.md)

## Problem Statement:

A common source of injury among weightlifters is incorrect form while performing exercises. The most popular strength exercises are the barbell bench press, squat, and deadlift. We seek to provide a system where a user can provide an input image/video of themselves performing one of these exercises, and the system would analyze the input and output recommendations on how to improve their form.

## Approach: 

The first part of our implementation is exercise determination: using human pose estimation (HPE) and action detection to figure out the exercise being performed in the input footage. This allows us to detect what are known as keypoints,  e.g. major body parts (arms, legs, spine, etc.) and joints (shoulders, ankles, knees, etc.). This is extremely relevant data for exercise applications.

Although there exist a few notable approaches to HPE, we will use the deep neural net method outlined by Cao et al. in Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields at Carnegie Mellon’s Perceptual Computing Lab. Cao et al. recommend two widely regarded keypoint models: MPII and Coco. We will use MPII since it does a better job of segmenting the torso and spine, which might prove useful in our analyses.

The next step of our implementation is to analyze the exercise form in the input footage to determine if form breakdown is occurring. First, we detect its keypoints, using either Chamfer distancing or normalized correlation with our reference exercise database to detect any deviations from ‘good’ form, isolating specific joints or body parts that are known to be areas of breakdown when it comes to a particular exercise. We can use the output of our human pose detection for both our input image and our good/bad form stored images to find similarities. Based on the chamfer distance or normalized correlation we calculate, we can advise the user on things they could work on.

It might also prove useful to use a Hough Transform to find potential curves in the pose detection output, since keeping the spine rigid is mostly an indicator of good form. For example, when a person is performing a deadlift, bending the back is very bad. However, during a bench press, a slight arch is actually optimal. By finding these areas of breakdown, or lack thereof, we can provide the user with targeted advice on how to improve their form.

### Example of Human Pose Detection:
<img src="assets\OpenPose.jpg" height="500px">

## Experiments and Results:

### Experimental Setup:

We will design our project to provide feedback to the user on 3 separate exercises: bench press, deadlift, and squat. The first step would be to identify the exercise the user is performing, and then give the user feedback on how to improve their form. To test the accuracy of our exercise detection and feedback, we will find 5 people of different size and gender performing the exercise correctly and incorrectly to find points of bias and better improve our program. 

### Data:

Datasets: [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
- Will be used to train our neural net model

Our Collected data:
We will collect images of good and bad form of our 3 different exercises from different angles to use as our base images that our input media will be compared to this [dataset](https://exrx.net/Lists/Directory)

### Code:

- We will utilize a code repository called OpenPose that utilizes a neural network model trained off a dataset of human pose points. This will help us detect the pose that our subject is doing and find points of interest.
  - [Tutorial](https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)
  - [Repository](https://github.com/spmallick/learnopencv/tree/master/OpenPose)
- We will implement Hough transform code to detect unwanted curves.
- We will implement the code that calculates the Chamfer distance between two human poses from the input media and stored image respectively to determine similarities and differences.
- We will implement the normalized correlation code that determines the similarities/difference between the input media and the stored images.
- Note: we will be comparing the effectiveness of Chamfer distance vs. normalized correlation to see what works better.

### Success Measure:

We can define our success in a series of milestones:
1. Successfully detect the appropriate joints and limbs in media where a person is doing either of our 3 exercises.
2. Successfully identify the exercise an individual is doing based on our human pose estimation on our media.
3. Successfully identify areas of bad form in an individual’s exercise performance.

## List of Experiments (perform with 5 people mentioned earlier):

One major point of uncertainty for this project is how our HPE model will interpret a barbell, which is required in the three exercises we are focusing on. The second is the subject angle or perspective of the user in the input footage. Most HPE experimentations done in the past rely on front-facing subjects. To capture an exercise’s area of potential form breakdown is not always possible with the camera focusing on users from the front. We will have to play around with our model to ensure that it is robust, working from multiple different angles per exercise.

### Identification of Exercise:

Provide several input images to our program to detect what exercises the input images are depicting. This experiment is crucial for our next 3 experiments. We will feed different variations of the 3 exercises with good and bad form to make sure we can detect them accurately.
- Uncertainties: If an exercise is performed egregiously poorly (with significant deviation from proper form), will our program still be able to accurately detect it?

### Deadlift:
- Good form: Attempt to detect features such as vertical arms, straight back.
- Bad form: Attempt to detect features such as curved back, diagonal or bent arms.

### Squat:
- Good Form: Attempt to detect features such as straight back, butt behind feet, hips parallel to ground.
- Bad Form: Attempt to detect features such as curved back, hips non-parallel to ground (signifies incomplete rep).

### Bench Press:
- Good Form: Attempt to detect tucked in elbows, slightly arched back, bar directly above chest
- Bad Form: Attempt to detect flat back, flared out elbows.
