# CS6910_Assignement_3
Use RNN to do tranliteration form hindi to hinglish

##Prerequisites
```
wget
python 3.12
pytorch 2.2.1
numpy 1.25
```
- I have conducted most some of my experments on Google collab. And most of them on PC by connecting Google collab to "local run time". If you run out of GPU time limit, connect with local run time.
- You need good GPU system to run the experiments on Collab. It will allow faster computation and training of your model.
- To run the code clone the repository, install wandb and wget which is used to import the dataset.

```
!pip install wget
pip install --upgrade pytorch
!pip install --upgrade wandb
```
- You can run the python code locally as well. for this install using following code
```
  pip install wandb
  pip install numpy
  pip install pytorch
```
- Just an FYI you need to have a decent GPU setup to run this on your PC.

## Dataset used for Experiments
- [Aksharantar dataset](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view) has been used for the experiments.

##Vanilla Seq2Seq
Followings are the Hyperparameters used in this part
| S. No.        | Hyperparameters       |  Values/types                                                    |
| ------------- | ----------------------|----------------------------- |
|1.             | Cell type             | RNN, LSTM, GRU          |
|2.             | Layer size            | [1,2,3,4]               |
|3.             | HIdden layer size     | [64, 128, 256]          |
|4.             | Dropout               | 0,0.2,0.4, 0.6          | 
|5.             | larning rate          | 0.001, 0.0001           |
  

The wandb report for this assignment can be found [here]().
## Author
[Vrijesh Kunwar](https://github.com/VrijKun)
ED23D015


