import numpy as np
from src import *
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


#pruebas para el random forest
#generamos explicativas aleatorias
train1=np.random.randn(50000,22)
test1=np.random.randn(50000,21)

try: 
        output1=modelo.modelo_random_forest(train1,test1,max_leaf=1)
        logging.info("el modelo se corrió con éxito")
except Exception as e:
        logging.exception("hubo un error con el modelo")


