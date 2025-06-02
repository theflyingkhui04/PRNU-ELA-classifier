def find_fake_real_files(files): #hàm phan loai anh trong file bang ten bat dau
    fake_files, real_files = 0, 0
    for file in files:
        filename = file.split("/")[-1]  # Lấy tên file từ đường dẫn
        if filename.startswith('red'):
            fake_files += 1
        if filename.startswith("Nikon"):
            real_files += 1
    return fake_files, real_files

