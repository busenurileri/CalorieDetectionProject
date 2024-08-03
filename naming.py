import os

def rename_images(folder_path, base_name):
    # Klasördeki tüm dosyaları al
    files = os.listdir(folder_path)
    
    # Sadece resim dosyalarını filtrele (jpg, jpeg, png)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Dosyaları sıralı bir şekilde yeniden adlandır
    for i, filename in enumerate(image_files, 1):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{base_name}{i}{file_extension}"
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")

# Kullanım
folder_path ='C:/Users/ileri/Downloads/images/hotdog' # Resimlerin bulunduğu klasörün yolu
base_name = 'hotdog'  # Yeni dosya adının başlangıç kısmı

rename_images(folder_path, base_name)
