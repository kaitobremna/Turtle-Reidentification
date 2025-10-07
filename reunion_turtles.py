import os
import json
import pandas as pd

class DatasetFactory:
    def __init__(self, root="."):
        self.root = root

    def finalize_catalogue(self, df):
        return df

class ReunionTurtles(DatasetFactory):
    summary = {
        'citation': 'Official Zenodo link: https://zenodo.org/record/7578241',
        'homepage': 'https://www.wildlife-datasets.com/reunion-turtles',
        'repository': 'https://github.com/WildMeOrg/wildlife-datasets/tree/main/reunionturtles',
        'description': 'A dataset of green sea turtles (Chelonia mydas) from Reunion Island, created for individual re-identification research.',
        'species': 'Green sea turtle (Chelonia mydas)',
    }
    archive = 'reunionturtles.zip'

    def __init__(self, root=".", download=False):
        super().__init__(root)
        if download:
            self._download()
            self._extract()
        self.df = self._create_catalogue()

    @classmethod
    def _download(cls):
        print("Downloading dataset...")
        pass

    @classmethod
    def _extract(cls):
        print("Extracting dataset...")
        pass

    def _create_catalogue(self) -> pd.DataFrame:
        annotations_path = os.path.join(self.root, 'annotations.json')
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"'{annotations_path}' not found. Please ensure 'annotations.json' is in the "
                f"correct directory and you have set the 'root' parameter correctly."
            )

        with open(annotations_path) as f:
            json_data = json.load(f)

        images_df = pd.DataFrame(json_data['images'])
        annotations_df = pd.DataFrame(json_data['annotations'])

        merged_df = pd.merge(images_df, annotations_df, left_on='id', right_on='image_id', suffixes=('_image', '_annotation'))

        df = pd.DataFrame({
            'image_id': merged_df['image_id'],
            'path': merged_df['path'],
            'identity': merged_df['identity'],
            'date': pd.to_datetime(merged_df['date'], format='%Y:%m:%d %H:%M:%S', errors='coerce'),
            'orientation': merged_df['position'],
            'species': 'Green sea turtle'
        })
        
        return self.finalize_catalogue(df)