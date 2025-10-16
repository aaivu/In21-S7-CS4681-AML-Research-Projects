class Config:
    def __init__(self):
        # Basic config
        self.is_training = 1
        self.model_id = 'Exchange_336_96'
        self.model = 'DLinear'
        self.n_ensemble = 5

        # Data loader
        self.data = 'custom'
        self.root_path = '../data/'
        self.data_path = 'exchange_rate.csv'
        self.features = 'M'
        self.target = 'OT' # This is not used for 'M' feature, but needs to be here.
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

        # Forecasting task
        self.seq_len = 336
        self.label_len = 48
        self.pred_len = 96

        # DLinear
        self.individual = False
        self.enc_in = 8 # Number of features in the dataset

        # Optimization
        self.num_workers = 0
        self.train_epochs = 10
        self.batch_size = 8
        self.patience = 3
        self.learning_rate = 0.0005
        self.loss = 'mse'
        self.lradj = 'type1'
        
        # GPU
        self.use_gpu = False
        self.gpu = 0

        self.confidence_level = 0.80

