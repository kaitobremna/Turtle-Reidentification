import torch
import numpy as np
from numpy.dtypes import UInt32DType
from wildlife_datasets.datasets import ReunionTurtles
from wildlife_tools.data.dataset import WildlifeDataset
import torchvision.transforms as T
from wildlife_tools.similarity.cosine import CosineSimilarity
import timm
from wildlife_tools.features.deep import DeepFeatures
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

# This block prevents the model loading error by trusting necessary numpy functions
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct, 
    np.ndarray, 
    np.dtype,
    UInt32DType
])

if __name__ == '__main__':
        
    filepath = "FILEPATHNAME"    #EDIT THIS TO WHERE YOU HAVE SAVED TURTLE FACES
    
    # Data Model and Set up
    print("Setting up dataset and model...")
    metadata = ReunionTurtles(filepath)
    name = 'hf-hub:BVRA/MegaDescriptor-B-224'
    transform = T.Compose([
        T.Resize([224, 224]), 
        T.ToTensor(), 
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Create true Open-set split
    print("Creating a master label map and a true open-set split...")
    all_unique_ids = metadata.df['identity'].unique()
    master_labels_map = {label: i for i, label in enumerate(all_unique_ids)}

    #Shuffled data
    shuffled_df = metadata.df.sample(frac=1, random_state=42).reset_index(drop=True)

    
    # Define the dataframes before creating the datasets
    database_df = metadata.df.iloc[100:,:]
    query_df = metadata.df.iloc[:100,:]
    
    dataset_database = WildlifeDataset(database_df, metadata.root, transform=transform)
    dataset_query = WildlifeDataset(query_df, metadata.root, transform=transform)

    extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True))

    # Feature Extraction
    print("Extracting features for the database...")
    database_features = extractor(dataset_database)
    
    # Check time to execute process, to be used for comparison against old methodology
    print("\nStarting timed query process")
    start_time = time.time()

    #Use cosine similarity to calculate a match probability from the features
    query_features = extractor(dataset_query)
    similarity_function = CosineSimilarity()
    similarity = similarity_function(query_features, database_features)['default']

    #Image must meet at least 0.75 similarity score to be considered a 'match'
    threshold = 0.75
    predictions = []
    database_labels = np.array(dataset_database.labels_string)
    for i in range(similarity.shape[0]):
        best_match_score = similarity[i].max()
        if best_match_score < threshold:
            predictions.append("new_individual")
        else:
            best_match_index = similarity[i].argmax()
            predictions.append(database_labels[best_match_index])
    predictions = np.array(predictions)
    end_time = time.time()
    
    # Evaluation and Results
    total_time = end_time - start_time
    num_queries = len(dataset_query)
    
    known_individuals = set(dataset_database.labels_string)
    open_set_ground_truth = np.array(
        [label if label in known_individuals else "new_individual" for label in dataset_query.labels_string]
    )
    accuracy = np.mean(open_set_ground_truth == predictions)

    print("\n--- Results ---")
    print(f"Total time for {num_queries} queries: {total_time:.2f} seconds")
    print(f"Average time per query: {total_time/num_queries:.4f} seconds")
    print("-----------------")
    print(f"Number of new individuals in query set: {np.sum(open_set_ground_truth == 'new_individual')}")
    print(f"Number of new individuals predicted: {np.sum(predictions == 'new_individual')}")
    print(f"Final Open-Set Accuracy: {accuracy:.4f}\n")

    # Visualisation, posting first 100 turtle images
    output_dir = "OUTPUTPATHNAME"    #EDIT TO YOUR FILEPATH    
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(100, len(query_df))):
        query_path = os.path.join(metadata.root, query_df.iloc[i]['path'])
        query_image = Image.open(query_path)
        true_label, predicted_label = open_set_ground_truth[i], predictions[i]

        if predicted_label != "new_individual":
            match_index = similarity[i].argmax()
            match_path = os.path.join(metadata.root, database_df.iloc[match_index]['path'])
            match_image = Image.open(match_path)
            match_score = similarity[i].max()
        else:
            match_image = Image.new('RGB', query_image.size, (255, 255, 255))
            match_score = similarity[i].max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(query_image); ax1.set_title(f"Query Image\nTrue ID: {true_label}"); ax1.axis('off')
        ax2.imshow(match_image); ax2.set_title(f"Best Match\nPredicted ID: {predicted_label}"); ax2.axis('off')
        
        is_correct = (true_label == predicted_label)
        fig.suptitle(f"Result for Query {i} - {'CORRECT' if is_correct else 'INCORRECT'}\n(Score: {match_score:.3f})",
                     fontsize=16, color='green' if is_correct else 'red')
        plt.savefig(os.path.join(output_dir, f"result_{i}.png")); plt.close()

    print(f"Saved {min(100, len(query_df))} comparison images to the '{output_dir}' directory.")
