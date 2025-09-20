Instructions to use this program:
1. Ensure you have a .env file contain your openrouter api key,
    your hivemq broker rul, port, user name, password.
    They should be in the form 
    OPENROUTER_API_KEY=xxxxxxxx
    HMQ_HOST=xxxxxxxxx.s1.eu.hivemq.cloud
    HMQ_PORT=8883
    HMQ_USER=xxxxx
    HMQ_PASS=xxxxx 
2. This project use a preload data stored in the data file as (sample_data_for_2label_test.json)for test you can change the  data in that file for your data. 
3. Your data structure should be formated as that in the test file

4. Run the main.py file to run the whole program 

Notes:
to use data from real sensor you just need to edit the sensor.py file but ensure that your data send by the sensor to backend is structured as the test file. 

the output of this is 
Ftype: the fault type detected by our fault type model
RUL: is the remaining useful life for the next fault detected by our rul model
Features text: the affects of the input features on the current output (the maximum three)
API: The api model response for the input features (fault type, maintenance action) 

This project does not train the models each time you call it. it just train them for the first time and save the models in the models file then call the pretrained models for prediction, it look at the models file for the models. if they are presented it just call them and if it does not find the model it go to train a new model and save it for the next call 