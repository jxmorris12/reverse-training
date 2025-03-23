import datasets


def _make_dbpedia_text(ex: dict[str, str]) -> dict[str, str]:
    ex["text"] = ex["title"] + " " + ex["content"]
    return ex

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
    def from_dataset_name(cls, dataset_name: str):
        """
        Create a ClassificationDataset from a named dataset.
        
        Args:
            dataset_name: The name of the dataset to load
            
        Returns:
            A ClassificationDataset object
        """
        # The dataset loading logic from load_dataset_from_name
        if dataset_name == "ag_news":
            ds = datasets.load_dataset("fancyzhx/ag_news")
            ds = ds["train"].train_test_split(test_size=0.1, seed=42)
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
            ds = ds["train"].train_test_split(test_size=0.1, seed=42)
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
            ds = ds.train_test_split(test_size=0.1, seed=42)
            ds["train"] = ds["train"].map(_make_dbpedia_text)
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "0": "Company",
                "1": "Organization",
                "2": "Location",
            }
        elif dataset_name.startswith("dbpedia_") and dataset_name[8:].isdigit():
            num_samples = int(dataset_name[8:])
            ds = datasets.load_dataset("fancyzhx/dbpedia_14")
            ds = ds.train_test_split(test_size=0.1, seed=42)
            ds["train"] = ds["train"].select(range(num_samples))
            text_column_name = "text"
            label_column_name = "label"
            label_map = {
                "0": "Company",
                "1": "Organization",
                "2": "Location",
            }
        elif dataset_name == "nq":
            ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
            ds = ds.train_test_split(test_size=0.1, seed=42)
            text_column_name = "text"
            label_column_name = None
            label_map = None
        elif dataset_name.startswith("nq_") and dataset_name[3:].isdigit():
            # Handle nq_BBB where BBB is any integer
            num_samples = int(dataset_name[3:])
            ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
            ds = ds.train_test_split(test_size=0.1, seed=42)
            ds["train"] = ds["train"].select(range(num_samples))
            text_column_name = "text"
            label_column_name = None
            label_map = None
        elif dataset_name == "msmarco":
            ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
            text_column_name = "text"
            label_column_name = None
            label_map = None
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")
        
        return cls(ds, text_column_name, label_column_name, label_map) 