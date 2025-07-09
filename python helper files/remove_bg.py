from PIL import Image

image = Image.open('C:/Users/padma/Downloads/training_data/coin-Photoroom.png').convert("RGBA")
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save('test.png', "PNG")  # Save as JPEG