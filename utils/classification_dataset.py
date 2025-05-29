import datasets


def _make_dbpedia_text(ex: dict[str, str]) -> dict[str, str]:
    ex["text"] = ex["title"] + " " + ex["content"]
    return ex


DBPEDIA_LABEL_MAP = {
    "A":"Company",
    "B":"Educational Institution",
    "C":"Artist",
    "D": "Athlete",
    "E": "Office Holder",
    "F": "Mean Of Transportation",
    "G": "Building",
    "H": "Natural Place",
    "I": "Village",
    "J": "Animal",
    "K": "Plant",
    "L": "Album",
    "M": "Film",
    "N": "Written Work",
}

NEWSGROUP_LABEL_MAP = {
    "A":"alt.atheism",
    "B":"comp.graphics",
    "C":"comp.os.ms-windows.misc",
    "D": "comp.sys.ibm.pc.hardware",
    "E": "comp.sys.mac.hardware",
    "F": "comp.windows.x",
    "G": "misc.forsale",
    "H": "rec.autos",
    "I": "rec.motorcycles",
    "J": "rec.sport.baseball",
    "K": "rec.sport.hockey",
    "L": "sci.crypt",
    "M": "sci.electronics",
    "N": "sci.med",
    "O": "sci.space",
    "P": "soc.religion.christian",
    "Q": "talk.politics.guns",
    "R": "talk.politics.mideast",
    "S": "talk.politics.misc",
    "T": "talk.religion.misc",
}

NUMBER_TO_LABEL = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T" ]


class ClassificationDataset:
    """
    A wrapper class for datasets.Dataset that includes metadata specific to classification tasks.
    """
    
    def __init__(
        self, 
        dataset: datasets.Dataset, 
        text_column_name: str, 
        label_column_name: str,
        label_map: dict = None
    ):
        """
        Initialize a ClassificationDataset.
        
        Args:
            dataset: The underlying datasets.Dataset object
            text_column_name: The name of the column containing text data
            label_column_name: The name of the column containing label data
            label_map: A dictionary mapping label values to human-readable names
        """
        self.dataset = dataset
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        
        # Use default AG News label map if none provided
        if label_map is None:
            self.label_map = {
                "A":"World",
                "B":"Sports",
                "C":"Business",
                "D": "Sci/Tech",
            }
        else:
            self.label_map = label_map
        
        self.dataset = self.dataset.cast_column(self.label_column_name, datasets.Value("int32"))
        self.dataset = self.dataset.map(self.relabel_example)
    
    def relabel_example(self, ex: dict[str, str]) -> dict[str, str]:
        """
        Relabel the example using the label map.
        """
        ex[self.label_column_name] = NUMBER_TO_LABEL[ex[self.label_column_name]]
        return ex
    
    def __getitem__(self, idx):
        """
        Access the underlying dataset with indexing.
        """
        return self.relabel_example(self.dataset[idx])
    
    def __len__(self):
        """
        Get the length of the underlying dataset.
        """
        return len(self.dataset)
    
    def select(self, *args, **kwargs):
        """
        Select examples from the underlying dataset.
        Returns a new ClassificationDataset with the selected examples.
        """
        selected_dataset = self.dataset.select(*args, **kwargs)
        return ClassificationDataset(
            selected_dataset, 
            self.text_column_name, 
            self.label_column_name,
            self.label_map
        )
    
    def train_test_split(self, *args, **kwargs):
        """
        Split the dataset into train and test sets.
        Returns a dictionary of ClassificationDataset objects.
        """
        split_datasets = self.dataset.train_test_split(*args, **kwargs)
        return {
            split: ClassificationDataset(
                dataset, 
                self.text_column_name, 
                self.label_column_name,
                self.label_map
            ) 
            for split, dataset in split_datasets.items()
        }
    
    @classmethod
    def from_dataset_name(cls, dataset_name: str, seed: int = 42):
        """
        Create a ClassificationDataset from a named dataset.
        
        Args:
            dataset_name: The name of the dataset to load
            seed: The seed to use for the train-test split
            
        Returns:
            A ClassificationDataset object
        """
        if " " in dataset_name:
            # Concatenate multiple datasets
            dataset_names = dataset_name.split(" ")
            dataset_list = [cls.from_dataset_name(name, seed) for name in dataset_names]
            ds = dataset_list[0].dataset
            for other_dataset in dataset_list[1:]:
                for split in other_dataset.dataset.keys():
                    # ds[split] = ds[split].concatenate(other_dataset.dataset[split])
                    ds[split] = datasets.concatenate_datasets([ds[split], other_dataset.dataset[split]])
            return cls(
                ds, 
                text_column_name=dataset_list[0].text_column_name, 
                label_column_name=dataset_list[0].label_column_name, 
            )

        if dataset_name == "ag_news":
            ds = datasets.load_dataset("fancyzhx/ag_news")
            ds = ds["train"].train_test_split(test_size=0.1, seed=seed)
            text_column_name = "text"        
            label_column_name = "label"
            label_map = {
                "A": "World",
                "B": "Sports",
                "C": "Business",
                "D": "Sci/Tech",
            }
        elif dataset_name.startswith("ag_news_") and dataset_name[8:].isdigit():
            num_samples = int(dataset_name[8:])
            ds = datasets.load_dataset("fancyzhx/ag_news")
            ds = ds["train"].train_test_split(test_size=0.1, seed=seed)
            assert len(ds["train"]) >= num_samples, f"Dataset {dataset_name} has only {len(ds['train'])} samples"
            ds["train"] = ds["train"].select(range(num_samples))
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "A": "World",
                "B": "Sports",
                "C": "Business",
                "D": "Sci/Tech",
            }
        elif dataset_name == "newsgroup":
            ds = datasets.load_dataset("SetFit/20_newsgroups")
            text_column_name = "text"
            label_column_name = "label"
            label_map = NEWSGROUP_LABEL_MAP
        elif dataset_name.startswith("newsgroup_") and dataset_name[10:].isdigit():
            num_samples = int(dataset_name[10:])
            ds = datasets.load_dataset("SetFit/20_newsgroups")
            ds["train"] = ds["train"].shuffle(seed=seed).select(range(num_samples))
            ds["test"] = ds["test"].shuffle(seed=42)
            text_column_name = "text"
            label_column_name = "label"
            label_map = NEWSGROUP_LABEL_MAP
        elif dataset_name == "rotten_tomatoes":
            ds = datasets.load_dataset("cornell-movie-review-data/rotten_tomatoes")
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "A":"neg",
                "B":"pos",
            }
        elif dataset_name.startswith("rotten_tomatoes_") and dataset_name[16:].isdigit():
            num_samples = int(dataset_name[16:])
            ds = datasets.load_dataset("cornell-movie-review-data/rotten_tomatoes")
            ds["train"] = ds["train"].shuffle(seed=seed).select(range(num_samples))
            ds["test"] = ds["test"].shuffle(seed=42)
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "A":"neg",
                "B":"pos",
            }
        elif dataset_name == "imdb":
            ds = datasets.load_dataset("stanfordnlp/imdb")
            ds["train"] = ds["train"].shuffle(seed=seed)
            ds["test"] = ds["test"].shuffle(seed=42)
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "A":"Negative",
                "B":"Positive",
            }
        elif dataset_name.startswith("imdb_") and dataset_name[5:].isdigit():
            num_samples = int(dataset_name[5:])
            ds = datasets.load_dataset("stanfordnlp/imdb")
            ds["train"] = ds["train"].shuffle(seed=seed).select(range(num_samples))
            ds["test"] = ds["test"].shuffle(seed=42)
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "A":"Negative",
                "B":"Positive",
            }   
        elif dataset_name == "dbpedia":
            ds = datasets.load_dataset("fancyzhx/dbpedia_14")
            ds = ds.map(_make_dbpedia_text)
            text_column_name = "text"
            label_column_name = "label"
            label_map = DBPEDIA_LABEL_MAP
        elif dataset_name.startswith("dbpedia_") and dataset_name[8:].isdigit():
            num_samples = int(dataset_name[8:])
            ds = datasets.load_dataset("fancyzhx/dbpedia_14")
            ds = ds.map(_make_dbpedia_text)
            ds["train"] = ds["train"].shuffle(seed=seed).select(range(num_samples))
            ds["test"] = ds["test"].shuffle(seed=seed)
            text_column_name = "text"
            label_column_name = "label"
            label_map = DBPEDIA_LABEL_MAP
        elif dataset_name == "nq":
            ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
            ds = ds.train_test_split(test_size=0.1, seed=seed)
            text_column_name = "text"
            label_column_name = None
            label_map = None
        elif dataset_name.startswith("nq_") and dataset_name[3:].isdigit():
            # Handle nq_BBB where BBB is any integer
            num_samples = int(dataset_name[3:])
            ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
            ds = ds.train_test_split(test_size=0.1, seed=seed)
            ds["train"] = ds["train"].select(range(num_samples))
            text_column_name = "text"
            label_column_name = None
            label_map = None
        elif dataset_name == "msmarco":
            ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
            text_column_name = "text"
            label_column_name = None
            label_map = None
        elif dataset_name.startswith("msmarco_") and dataset_name[8:].isdigit():
            num_samples = int(dataset_name[8:])
            ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")["train"]
            ds = ds.train_test_split(test_size=0.1, seed=seed)
            ds["train"] = ds["train"].select(range(num_samples))
            text_column_name = "text"
            label_column_name = None
            label_map = None
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")
        
        return cls(ds, text_column_name, label_column_name, label_map) 
