### Saket Shirsath, Parth Thakkar, Ben Wolfson
<br>
[Project Proposal](proposal.md)


## Abstract:

Weightlifters, both old and new injure, themselves while performing popular exercises like bench press, squats, and deadlifts. We aim to create a program that will identify which exercise is being performed and what corrections need to be made to have ideal form. Our main objective for this update is to identify which exercise is being performed. We took two approaches to identify exercises: a convolutional neural network pre-trained with the MPII Human Pose Dataset and a convolutional neural network built with Keras and trained with images scraped from the internet. Both provided very promising initial results.

## Teaser Figure:

<img src="assets\teaser.png" height="500px">


## Pose Identification using a Convolutional Neural Network: 

One of our milestones for this iteration of our project was the successful pose detection of the exercises we are working with. This entails finding the wire figures of a person’s body when performing a particular exercise. The approach we take with this requires us to train a neural network with the MPII Human Pose Dataset. The first stage of the process is to create a set of 2D confidence maps of body part locations such as the elbow, knee, wrist, etc. Once that is done, the confidence maps are run through an algorithm to produce the 2D joint locations for the person in the image. 

| Squat | Bench | Deadlift |
| ----- | ----- | -------- |
|<img src="assets\squat_pose.png" height="500px">|<img src="assets\bench_pose.png" height="500px">|<img src="assets\deadlift_pose.png" height="500px">|


## Results for HPE

Our goal while using this approach was to develop a library of poses for each of these exercises so that when we pass in an input image into our program, we can compare the pose for that to our library using an image comparison algorithm such as shortest squared distance or structural similarity measure. We quickly realized that while this approach was not very consistent. There were often similarities in pose position when it came to exercises like the squat and the deadlift, and we were not getting accurate results. It seemed that the position of the barbell with respect to the body in the image was a bigger factor in exercise classification than we originally thought.

Nevertheless, the results of the human pose estimation were very promising. We speculate that the actual positions of the joints could definitely be taken into account to give form recommendations to the user if we would classify the exercise being performed correctly. As a result, we needed to find a more accurate way to detect our exercises.

## Image Classification with Categorical Convolutional Neural Network

To solve our issue with accurate exercise classification, we experimented with a brute-force categorical convolutional neural network built with Keras. Using the Bing Web Search API, we scraped 250 images for each of the chosen exercises: barbell bench press, barbell back squat, and barbell deadlift. Then, we manually filtered through those to throw away faulty representations and fix formatting issues. For our final dataset, we were left with 155 bench press images, 196 squat images, and 198 deadlift images. For model validation, we randomly allocated 20% of our dataset. The associated image collections are shown in the Appendix.



