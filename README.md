Tensorflow 2.0 suite for Training and Transfer-Learning

# Params

--mode: "train" or "predict"

**--dataset: folder containing 2 folders (train, val) which have other folders with the class names, containing images to be trained for the model**

--model: parameter that defines the network to be trained and can be: "InceptionV3", "MobileNet", "MobileNetV2", "InceptionResNetV2", "ResNet", "NasNet", "VGG16", "VGG19", "Xception"

**--optimizer: type of network optimizer, which can be: Adam, Adagrad, RMSprop and sgd**

--image_size: dimension of the image / network input.

--lr: model training learning rate (default: 0.0001)

--batch_size: size of model training lots (default: 32)

--epoch_steps: number of steps taken per training season (default: 500)

--val_steps: number of steps performed by validation (default: 50)

--num_epoch: number of seasons to be trained (default: 10)

--output_dir: Directory where training models and checkpoints will be saved

--validate_every: defines the model to be validated and saved every time (checkpoint). (default: 1)

--lr_decay: parameter to define whether learning rate decay will be used, in order to improve the convergence of the model over time (default: False).

--dropout: Number of layers (from 0 to 1) to be turned off at random to mitigate overfitting. (default: 0.4)

--retrain_hdf5: Path to hdf5 file containing model and weights, for transfer-learning.

--finetune: Parameter that defines whether the transfer learning applied under the loaded model of retrain_hdf5 is about models with different number of classes (default: False)

**Mandatory parameters are marked in bold**

# Example:

**python train_model.py --dataset ./dataset_folder/ --batch_size 32 --epoch_steps 500 --val_steps 50 --num_epoch 20 --lr 0.0001**
