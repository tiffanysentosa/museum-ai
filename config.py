import os


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.GEMINI_API_KEY = "default_api_key"
            cls._instance.DATASET_PATH = "metdata.json"
            cls._instance.BASE_URL = (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            cls._instance.MODEL = "gemini-1.5-flash"
            cls._instance.TOKENIZER = None  # Placeholder for tokenizer
            cls._instance.CLIENT = None
        return cls._instance

    def update_config(
        self,
        api_key=None,
        dataset_path=None,
        base_url=None,
        model=None,
        tokenizer=None,
        CLIENT=None,
    ):
        if api_key:
            self.GEMINI_API_KEY = api_key
        if dataset_path:
            self.DATASET_PATH = dataset_path
        if base_url:
            self.BASE_URL = base_url
        if model:
            self.MODEL = model
        if tokenizer:
            self.TOKENIZER = tokenizer
        if CLIENT:
            self.CLIENT = CLIENT
