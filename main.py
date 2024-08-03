from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import shutil
import random
import yaml

def create_yolo_format():
    """
    Creates a folder structure suitable for YOLO format.
    """
    print("Creates a folder structure suitable for YOLO format....")
    Path('yolo_format/images/train').mkdir(parents=True, exist_ok=True)
    Path('yolo_format/images/val').mkdir(parents=True, exist_ok=True)
    Path('yolo_format/labels/train').mkdir(parents=True, exist_ok=True)
    Path('yolo_format/labels/val').mkdir(parents=True, exist_ok=True)
    print("Folder structure created.")

def split_data(images_dir, labels_dir, train_ratio=0.8):
    """
    Separates the data into training and validation sets and copies them into the appropriate folders.
    """
    create_yolo_format()
    all_images = list(Path(images_dir).glob('*.*'))
    random.shuffle(all_images)
    split_index = int(len(all_images) * train_ratio)
    
    print("Data is being separated into training and validation sets...")
    for i, img_path in enumerate(all_images):
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        if i < split_index:
            shutil.copy(img_path, 'yolo_format/images/train')
            shutil.copy(label_path, 'yolo_format/labels/train')
        else:
            shutil.copy(img_path, 'yolo_format/images/val')
            shutil.copy(label_path, 'yolo_format/labels/val')
    print("Data separation is completed.")

def create_yaml():
    """
    Creates a YAML file containing the training data and class names.
    """
    data = {
        'train': str(Path('yolo_format/images/train').resolve()),
        'val': str(Path('yolo_format/images/val').resolve()),
        'nc': 7,
        'names': ['pizza', 'burger', 'friedpatato', 'nugget', 'cola', 'hotdog', 'onionring']
    }
    print("Creates a YAML file...")
    with open('fastfood.yaml', 'w') as file:
        yaml.dump(data, file)
    print("YAML file is created.")

def train_model():
    """
    Trains the YOLOv8 model with the training data.
    """
    print("YOLOv8 model is being loaded and training is starting...")
    model = YOLO("yolov8n.yaml").load("yolov8n.pt") #Configure from YAML and transfer weights
    results = model.train(data="fastfood.yaml", epochs=50, imgsz=640)
    print("Model training completed.")

def predict_calories(model_path, input_folder, output_folder):
    """
    Loads the trained model, performs calorie estimation on all images in the specified folder, 
    and saves the results to the specified output folder.

    :param model_path: Path to the trained model file
    :param input_folder: Folder containing the images for prediction
    :param output_folder: Folder where result images will be saved

    """
    print("Loading model..")
    try:
        model = YOLO(model_path)
        print("Model successfully loaded.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    calorie_dict = {
        'pizza': 285,
        'burger': 250,
        'friedpatato': 365,
        'nugget': 300,
        'cola': 150,
        'hotdog': 150,
        'onionring': 200
    }

    print(f"Calorie predictions are being made... Input folder: {input_folder}")
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob('*.*'))
    print(f"Found image files: {len(image_files)}")

    if not image_files:
        print("Image file not found")
        return

    for image_path in image_files:
        print(f"\nMaking predictions for {image_path}...")
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            results = model(img)
            if results is None or len(results) == 0:
                print(f"Results could not be obtained or are empty: {image_path}")
                continue

            total_calories = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    c = box.cls
                    class_name = model.names[int(c)]
                    calories = calorie_dict.get(class_name, 0)
                    total_calories += calories
                    print(f'Detected: {class_name}: {calories} kcal')

                    #Draw bounding box and add label
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'{class_name}: {calories} kcal'
                    cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print(f"Total calories: {total_calories} kcal")

            #Save the result image
            output_file = output_path / f'result_{image_path.name}'
            cv2.imwrite(str(output_file), img)
            print(f'Prediction result saved as {output_file}.')

        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

    print("Calorie predictions completed.")

def predict_and_save_image(image_path, model_path, output_dir):
    """
    Takes the specified image, makes predictions, and saves the result to the specified folder

    :param image_path: Path to the image file for prediction
    :param model_path: Path to the trained model file
    :param output_dir: Folder where the result image will be saved

    """
    print("Loading model...")
    try:
        model = YOLO(model_path)
        print("Model successfully loaded.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    calorie_dict = {
        'pizza': 285,
        'burger': 250,
        'friedpatato': 365,
        'nugget': 300,
        'cola': 150,
        'hotdog': 150,
        'onionring': 200
    }

    print(f"Making predictions for {image_path}...")
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image: {image_path}")
            return

        results = model(img)
        if results is None or len(results) == 0:
            print(f"Results could not be obtained or are empty: {image_path}")
            return

        total_calories = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                class_name = model.names[int(c)]
                calories = calorie_dict.get(class_name, 0)
                total_calories += calories
                print(f'Detected: {class_name}: {calories} kcal')

                #Drawing bounding box and adding label
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{class_name}: {calories} kcal'
                cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Total calories: {total_calories} kcal")

        #"Save the result image"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f'result_{Path(image_path).name}'
        cv2.imwrite(str(output_file), img)
        print(f'Prediction result saved as {output_file}.')

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

    print("Prediction process completed.")

def main():
    images_dir = 'images'
    labels_dir = 'labels'
    
    
    #split_data(images_dir, labels_dir)
    
   
    #create_yaml()
    
    
    #train_model()

   
    model_path = "runs/detect/train/weights/best.pt"  # Eğitilmiş model yolu
   # predict_calories(model_path,"images","results")

    single_image_path = "fastfood3.png" # Tek resim yolu
    predict_and_save_image(single_image_path,model_path,"results")

if __name__ == '__main__':
    main()
