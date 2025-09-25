from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(filename, text):
    # Create a new image with a gradient background
    img = Image.new('RGB', (800, 600), color='#4f46e5')
    d = ImageDraw.Draw(img)
    
    # Add text
    d.text((400, 300), text, fill='white', anchor='mm')
    
    # Save the image
    img.save(os.path.join('static', 'carousel', filename))

# Create the directory if it doesn't exist
os.makedirs(os.path.join('static', 'carousel'), exist_ok=True)

# Create placeholder images
for i in range(1, 5):
    create_placeholder(f'p{i}.jpg', f'Sample Image {i}')

for i in range(1, 6):
    create_placeholder(f'men{i}.JPG', f'Men\'s Style {i}')
    create_placeholder(f'women{i}.JPG', f'Women\'s Style {i}')

create_placeholder('banner.JPG', 'Welcome Banner')
create_placeholder('loadmore.JPG', 'Load More')
create_placeholder('loader.gif', 'Loading...')