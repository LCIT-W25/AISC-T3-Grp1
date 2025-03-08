import pickle as pkl
import datetime as dt

class Pickle:
    
    def __init__(self) -> None:
        pass # will create the class instance

    pkl_const = '.pkl'

    def save_model_to_pickle_folder(self, model : any, model_name : str):

        file_path = 'PickledModels\\' +  model_name +'_'+ str(dt.datetime.now().strftime("%d%m%Y%H%M%S")) +self.pkl_const
        pkl.dump(model , open(file_path, 'wb'))
        print(f'Model Saved Successfully as {file_path}')

    def load_model_from_pickle_folder(self, file_name : str):

        file_path = r'PickledModels/'+file_name+self.pkl_const
        model = pkl.load(open(file_path , 'rb'))
        print(f'Model loaded successfully from {file_path}')
        return model
    
    def save_model_to_specified_path(self, model : any, path : str, model_name : str):

        file_path = r''+path+'\\' + model_name +'_'+ str(dt.datetime.now().strftime("%d%m%Y%H%M%S")) +self.pkl_const
        pkl.dump(model , open(file_path, 'wb'))
        print(f'Model Saved Successfully as {file_path}')

    def load_model_from_specified_path(self, path : str, file_name : str):

        file_path = r''+path+'/'+file_name+self.pkl_const
        model = pkl.load(open(file_path, 'rb'))
        print(f'Model loaded successfully from {file_path}')
        return model