## Implementation of Decision Tree using ID3 Algorithm:

![Entropy_Formula](entropy.PNG) 
#####  Where Pi is the proportion of class i (with C being all classes in the data set).
#### Task: Use Information Gain as your decision measure and treat all features as discrete multinomial distributions.Given are the two data sets named car and nursery as csv files. Your program should be able to read both data sets and treat the last value of each line as the class. Your task is to correctly implement the ID3 algorithm and return the final tree without stopping early (both data sets can be learned perfectly, i.e. all leaves have an entropy of 0). The output of your algorithm should look like the example XML solution given for the car data set. With that, you can check the correctness of your solution. All features are unnamed on purpose, please number them according to the column starting from 0 (e.g. att0). Machine learning libraries are not allowed. You can use libraries for handling the XML format and the input parameters. Your program must accept the following parameters:

#### 1. data - The location of the data file (e.g. /media/data/car.csv).
#### 2. output - Where to write the XML solution to (e.g. /media/data/car solution.xml).



### Example statement to execute the program on the terminal: python decisiontree.py --data nursery.csv --output car_output.xml
