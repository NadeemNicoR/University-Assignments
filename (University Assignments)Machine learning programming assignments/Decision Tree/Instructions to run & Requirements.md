



# Instructions to run the code
======

## PythonDescisonTree
======
#### Install requirements using the below command

#### pip install -r requirements.txt

#### Running the script
#### The code can use any csv file and create a decision tree based on the data in the csv. The script supports the below options data output header

#### data: The path of the CSV file to use for creation of the decision tree.
#### output: The output of the script (decision tree) gets stored into a XML file. Thus file path were the tree must be saved is given in this option
#### header: If the data contains exactly first row as the headers and first column as row IDs specify it as True while execution of script (optional)


### sample : python DecisionTreeSolution.py --data car.csv --output car.xml


# Requirements:
======
### pandas
### xmltodict
