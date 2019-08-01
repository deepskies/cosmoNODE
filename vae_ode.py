# using Yulia Rubanova's latent_ode repo
# i think this will go into parse_datasets():
# mimic of class PersonActivity(object):

# in run_models.py this will be a dataset_obj

class FluxVAE(object):
    # tag_ids is similar to passband imo

    # label_names = targets -> dictionary
    def __init__(self, root, dl=False, reduce='average', max_seq_length=350,
                n_samples=None, device=torch.device('cpu')):
        df = pd.read_csv('./demos/data/training_set.csv')
        meta = pd.read_csv('./demos/data/training_set_metadata.csv')
        labels = label_dict(meta, 'target')
        labels.append(99)  # dummy unknown class

        self.data_len = len(self.data)
        self.n_labels = len(labels)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return

def label_dict(self, df, target_col):
    labels = sorted(df[target_col].unique())
    return labels

if dataset_name == 'classify_lc':
    n_samples = __
    dataset_obj = FluxVAE('file path here', dl=False, n_samples=n_samples, device=device)

    tr_data, te_data = train_test_split(dataset_obj)

    train_data = []
