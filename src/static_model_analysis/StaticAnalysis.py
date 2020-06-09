import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))


from model.MetaModel import  *




def test_residual_ratio():
    hyperparameters = Hyperparameters()


    model = MetaModel(hyperparameters)
    model.populate_from_embedding(MetaModel.get_nasnet_embedding())
    model.cells[0].process_stuff()
    model.cells[1].process_stuff()


if __name__ == '__main__':
    test_residual_ratio()