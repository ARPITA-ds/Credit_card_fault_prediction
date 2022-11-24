from flask import Flask
from CreditCard.logger import logging
from CreditCard.Exception import CreditException
import sys

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    try:
        raise Exception("we are testing custom Exception")

    except Exception as e:
        Credit = CreditException(e,sys)
        logging.info(Credit.error_message)
        logging.info("Testing logging module")
    return "Starting Credit card fault prediction"


if __name__=="__main__":
    app.run(debug=True)