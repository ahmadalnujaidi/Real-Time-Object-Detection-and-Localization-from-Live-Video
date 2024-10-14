# download model from Google Drive using gdown
import os


# Install necessary dependencies
os.system('pip install gdown ultralytics')

# Use gdown to download the ONNX model from Google Drive
import gdown
url = "https://drive.google.com/uc?id=1KnW1maJrQnpowjgJdYxykXjNzZyH6cUW"
output = "model.onnx"
gdown.download(url, output, quiet=False)

print("Model downloaded successfully.")

# -----------------------------------------------------------

# competition video URL:
# https://drive.google.com/drive/folders/1OUr-edxMZkR1-j7NEDwJ36Ye_GCKmxoc
# test vid 1 id= 1BJDHWae88l4GFlT4DnZZvKJ8NBJ_Zoht
# test vid 2 id= 1Q9DfAffOCGsE-w3VAU7OfcyNhmb8UtP3 
# test vid 3 id= 1zVwNyH8MKc9Cq7FroySxGcRelMelqflu
# test vid 4 id= 15cAS8sKDRfi4WnDnseusXamXwVpG9m5u
# test vid 5 id= 1YDHLv15wPJx0dbsBj-6ED83FDQnedqPx

# the longer the video the longer it takes to execute
# Replace 'your_file_id' with the file ID from the Google Drive link
# url= 'https://drive.google.com/uc?id="your_file_id"' e.g testvid1 id=1BJDHWae88l4GFlT4DnZZvKJ8NBJ_Zoht
url = 'https://drive.google.com/uc?id=1zVwNyH8MKc9Cq7FroySxGcRelMelqflu'
output = 'testing_video.mp4'  # Local filename

# Download the video
gdown.download(url, output, quiet=False)
print("Video downloaded successfully.")