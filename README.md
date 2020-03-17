## Gender Prediction-Flask-Deployment
This web app predcits Gender by taking height and weight as the inputs.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict Gender of the individual based on training data in 'weight_height.csv' file.
2. app.py - This contains Flask APIs that receives Individual details like weight and height through GUI or API calls, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.
4. static -This folder contains the CSS styles.
### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

Enter valid numerical values in all 2 input boxes and hit Predict.
