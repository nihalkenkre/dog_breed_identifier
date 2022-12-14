# Dog Breed Identifier

A dog identifier webapp which asks the user for an image and tries to identify the breed od the dog present in the image. If the image contains a human, it will try to detect the breed which resembles the human.


Example 1
![identify_1](imgs/identify_1.png)

Example 2
![identify_2](imgs/identify_2.png)

## Flow

The app looks for a human in the image, if a human is detected, the app looks for the resembling dog breed. If a dog is detected then the app looks for the dog breed.

The app uses openCV's human detection workflow to detect the presence of humans in the provided image. 
The app uses a custom CNN with learning transferred from the Resnet 50 model to identify the dog breed. The last two layers of the Resnet50 model are dropped off and a custom model is added to identify the breeds.

## Commands
Run the following from the project folder

`python ./app/run.py`