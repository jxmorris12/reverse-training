import datasets


def _make_dbpedia_text(ex: dict[str, str]) -> dict[str, str]:
    ex["text"] = ex["title"] + " " + ex["content"]
    return ex

DBPEDIA_LABEL_MAP = {
    "0": "Company",
    "1": "Educational Institution",
    "2": "Artist",
    "3": "Athlete",
    "4": "Office Holder",
    "5": "Mean Of Transportation",
    "6": "Building",
    "7": "Natural Place",
    "8": "Village",
    "9": "Animal",
    "10": "Plant",
    "11": "Album",
    "12": "Film",
    "13": "Written Work",
}
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
                "0": "World",
                "1": "Sports",
                "2": "Business",
                "3": "Sci/Tech",
            }
        else:
            self.label_map = label_map
    
    def __getitem__(self, idx):
        """
        Access the underlying dataset with indexing.
        """
        return self.dataset[idx]
    
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
                label_map=dataset_list[0].label_map
            )

        if dataset_name == "ag_news":
            ds = datasets.load_dataset("fancyzhx/ag_news")
            ds = ds["train"].train_test_split(test_size=0.1, seed=seed)
            text_column_name = "text"        
            label_column_name = "label"
            label_map = {
                "0": "World",
                "1": "Sports",
                "2": "Business",
                "3": "Sci/Tech",
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
                "0": "World",
                "1": "Sports",
                "2": "Business",
                "3": "Sci/Tech",
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