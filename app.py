from flask import Flask
from CreditCard.logger import logging

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():

    logging.info("Testing logging module")
    return "Starting Credit card fault prediction"


if __name__=="__main__":
    app.run(debug=True)