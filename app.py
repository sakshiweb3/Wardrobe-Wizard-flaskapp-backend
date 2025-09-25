from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
import os
from PIL import Image
from script import predict
import numpy as np
from utils.evaluate import execute
from pose_parser import pose_parse
from services.description import describe_outfit
import time
from services.ai_fashion import (
    get_style_recommendation,
    analyze_outfit_compatibility,
    get_outfit_suggestions,
    get_fashion_tips,
    analyze_current_outfit
)

# Initialize Flask app and global variables
app = Flask(__name__, 
          static_url_path='', 
          static_folder='../frontend/static',
          template_folder='../frontend/templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
cascPath = "./models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_stream = None
flag = True
selection = ""

# Ensure required directories exist
os.makedirs(os.path.join('static', 'captures'), exist_ok=True)
os.makedirs(os.path.join('static', 'images', 'clothing'), exist_ok=True)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not self.video.isOpened():
            raise ValueError("Could not open camera")
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        
        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)
        
        # Convert to jpg format
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes() if ret else None

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/wardrobe')
def wardrobe():
    return render_template('wardrobe.html')

@app.route('/try-on')
def try_on():
    # Get all available clothing images
    clothing_dir = os.path.join(app.static_folder, 'images', 'clothing')
    available_clothes = []
    if os.path.exists(clothing_dir):
        available_clothes = [f for f in os.listdir(clothing_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('try_on.html', clothes=available_clothes)

@app.route('/try-on/<int:outfit_id>', methods=['POST'])
def process_try_on(outfit_id):
    try:
        # Get the user's photo
        user_photo_path = os.path.join(app.static_folder, 'captures', 'temp_capture.jpg')
        
        if not os.path.exists(user_photo_path):
            return jsonify({
                'success': False,
                'error': 'No photo found. Please take a photo first.'
            })

        # Process the virtual try-on using your existing model
        result_path = execute(user_photo_path, str(outfit_id))
        
        if result_path and os.path.exists(result_path):
            # Convert the result path to a URL
            result_url = url_for('static', filename=os.path.relpath(result_path, app.static_folder))
            return jsonify({
                'success': True,
                'result_image': result_url
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate try-on result.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get the base64 image data from the request
        data = request.json['image']
        # Remove the data URL prefix
        img_data = data.split(',')[1] if ',' in data else data
        
        # Convert base64 to image
        import base64
        from io import BytesIO
        
        img = Image.open(BytesIO(base64.b64decode(img_data)))
        
        # Save the image
        captured_path = os.path.join('static', 'captures', 'temp_capture.jpg')
        os.makedirs(os.path.dirname(captured_path), exist_ok=True)
        img.save(captured_path)
        
        return jsonify({
            'success': True,
            'redirect': url_for('try_on', image_path=captured_path)
        })
    except Exception as e:
        print(f"Error capturing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Create required directories for virtual try-on functionality only
os.makedirs("./Database/val/person", exist_ok=True)
os.makedirs("./output/second/TOM/val", exist_ok=True)
os.makedirs("./static/images/clothing", exist_ok=True)  # Only for try-on samples

# Clusters are only used for the virtual try-on feature
# They are not displayed on the welcome page anymore

# Define clothing categories and their mapping
clothing_categories = {
    'men': {
        'casual': ['0', '1', '2'],
        'formal': ['3', '4'],
        'ethnic': ['5']
    },
    'women': {
        'dresses': ['6'],
        'tops': ['7'],
        'ethnic': ['8', '9']
    }
}

# Sample product details
product_details = {
    # Men's Collection
    '0': {'name': 'Classic Fit Polo Shirt', 'price': 999, 'category': 'casual', 'brand': 'HalfCut'},
    '1': {'name': 'Slim Fit Denim Shirt', 'price': 1299, 'category': 'casual', 'brand': 'Denim Co'},
    '2': {'name': 'Athletic Fit Sports Tee', 'price': 849, 'category': 'casual', 'brand': 'ActivePro'},
    '3': {'name': 'Formal Business Shirt', 'price': 1499, 'category': 'formal', 'brand': 'Executive'},
    '4': {'name': 'Premium Wool Blazer', 'price': 3999, 'category': 'formal', 'brand': 'Luxe'},
    '5': {'name': 'Designer Kurta Set', 'price': 2499, 'category': 'ethnic', 'brand': 'Ethnix'},
    
    # Women's Collection
    '6': {'name': 'HalfCut Black Tees', 'price': 849, 'category': 'tops', 'brand': 'HalfCut'},
    '7': {'name': 'Full Sleeve Deep Neck Top', 'price': 649, 'category': 'tops', 'brand': 'Trendy'},
    '8': {'name': 'Adidas Blocked Crop Top', 'price': 949, 'category': 'tops', 'brand': 'Adidas'},
    '9': {'name': "Women's Black Sleeveless Top", 'price': 749, 'category': 'tops', 'brand': 'Fashion'},
    '10': {'name': 'Stone Washed Wide Neck Top', 'price': 799, 'category': 'tops', 'brand': 'Urban'},
    '11': {'name': 'Reebok Peach Fleeted Top', 'price': 949, 'category': 'tops', 'brand': 'Reebok'}
}

# Get images for a specific category
def get_category_images(category, gender='men'):
    images = []
    categories = clothing_categories[gender]
    if category in categories:
        for cluster in categories[category]:
            cluster_path = f"./clusters/{cluster}"
            if os.path.exists(cluster_path):
                for img in os.listdir(cluster_path)[:1]:
                    images.append({
                        'path': f"clusters/{cluster}/{img}",
                        'details': product_details[cluster]
                    })
    return images

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not self.video.isOpened():
            print("Error: Could not open camera")
            self.video = None
        else:
            # Set resolution
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        if self.video:
            self.video.release()

    def get_frame(self):
        if not self.video or not self.video.isOpened():
            # Return a blank frame if camera is not available
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank_frame, True

        ret, frame = self.video.read()
        if not ret or frame is None:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank_frame, True

        # Make sure frame is not too small
        if frame.shape[0] < 433 or frame.shape[1] < 465:
            frame = cv2.resize(frame, (640, 480))

        # Safely crop the frame
        try:
            cropped_frame = frame[48:433, 176:465]
            img = cv2.resize(cropped_frame, (192, 256))
            cv2.imwrite('temp.jpg', img)
        except Exception as e:
            print(f"Error cropping frame: {e}")
            cropped_frame = frame  # Use full frame if cropping fails

        try:
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
        except Exception as e:
            print(f"Error detecting faces: {e}")
            faces = []

        err = len(faces) == 0
        for (x, y, w, h) in faces:
            try:
                face_img = cropped_frame[max(0, y-10):min(cropped_frame.shape[0], y+h+10), 
                                      max(0, x):min(cropped_frame.shape[1], x+w)]
                cv2.imwrite('face.jpg', face_img)
                cv2.rectangle(cropped_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            except Exception as e:
                print(f"Error saving face: {e}")

        return cropped_frame, err

@app.route('/')
def index():
    global flag
    flag = True
    
    # Get all clothing images
    clothing_dir = os.path.join(app.static_folder, 'images', 'clothing')
    clothing_images = []
    
    if os.path.exists(clothing_dir):
        for file in os.listdir(clothing_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                clothing_images.append({
                    'src': url_for('static', filename=f'images/clothing/{file}'),
                    'name': os.path.splitext(file)[0].replace('_', ' ').title()
                })
    
    # If no clothing images found, use cluster images as fallback
    if not clothing_images:
        for cluster in range(10):
            cluster_path = f"./clusters/{cluster}"
            if os.path.exists(cluster_path):
                for img in os.listdir(cluster_path)[:1]:  # Get first image from each cluster
                    clothing_images.append({
                        'src': f"clusters/{cluster}/{img}",
                        'name': f'Style {cluster + 1}'
                    })
    
    return render_template('index.html', 
                         clothing_images=clothing_images,
                         os=os,
                         description=describe_outfit)

# Old routes removed to avoid duplication

@app.route('/cast', methods=['POST'])
def cast():
    global selection
    selection = request.form["selection"]
    
    # Get clothing image path
    clothing_path = f"./static/images/clothing_{int(selection) + 1}.jpg"
    if not os.path.exists(clothing_path):
        return "Selected clothing not found", 404
        
    # Get AI style recommendations for the selected item
    recommendations = get_style_recommendation(f"Outfit {int(selection) + 1}")
    compatibility = analyze_outfit_compatibility(f"Outfit {int(selection) + 1}", "temp.jpg")
    
    return render_template('final.html', 
                         sel=selection,
                         recommendations=recommendations,
                         compatibility=compatibility,
                         clothing_path=f"images/clothing_{int(selection) + 1}.jpg")

def gen(camera):
    timer = 100
    global video_stream
    try:
        while timer > 0:
            frame, err = camera.get_frame()
            if frame is None:
                continue
                
            # Add timer text
            cv2.putText(frame, str(timer//10), (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
            
            # Add guide overlay
            h, w = frame.shape[:2]
            cv2.putText(frame, "Position your face in the center", (w//4, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Convert frame to JPEG
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    if err != True:
                        timer -= 1
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
                
    except Exception as e:
        print(f"Error in gen: {e}")
    finally:
        if video_stream:
            video_stream.__del__()

def gen_stored(path):
    try:
        img = cv2.imread(path)
        if img is None:
            # Return a blank frame if image can't be loaded
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No image available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', img)
        if ret:
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(f"Error in gen_stored: {e}")

@app.route('/video_feed')
def video_feed():
    global video_stream, flag
    try:
        if flag:
            video_stream = VideoCamera()
            if video_stream.video and video_stream.video.isOpened():
                flag = False
                return Response(gen(video_stream), 
                              mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                print("Failed to open camera")
                return "Camera not available", 503
        else:
            return Response(gen_stored("temp.jpg"), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return "Error accessing camera", 503

@app.route('/men')
def men():
    # Get products by category for men
    casual_wear = get_category_images('casual', 'men')
    formal_wear = get_category_images('formal', 'men')
    ethnic_wear = get_category_images('ethnic', 'men')
    
    return render_template('men.html', 
                         casual_wear=casual_wear,
                         formal_wear=formal_wear,
                         ethnic_wear=ethnic_wear)

@app.route('/women')
def women():
    # Get products by category for women
    dresses = get_category_images('dresses', 'women')
    tops = get_category_images('tops', 'women')
    ethnic_wear = get_category_images('ethnic', 'women')
    
    return render_template('women.html',
                         dresses=dresses,
                         tops=tops,
                         ethnic_wear=ethnic_wear)

@app.route('/final_img')
def final_img():
    global selection
    timestamp = str(int(time.time()))
    
    try:
        # Create necessary directories if they don't exist
        os.makedirs("./Database/val/person", exist_ok=True)
        os.makedirs("./output/second/TOM/val", exist_ok=True)
        
        # Save the captured image
        person = Image.open('temp.jpg')
        person_path = f"./Database/val/person/{timestamp}.jpg"
        person.save(person_path)
        
        # Process the image
        pose_parse(timestamp)
        execute()
        
        # Create the try-on pair
        with open("./Database/val_pairs.txt", "w") as f:
            f.write(f"{timestamp}.jpg {selection}_1.jpg")
        
        # Generate the try-on result
        predict()
        
        # Process the result image
        result_path = f"./output/second/TOM/val/{selection}_1.jpg"
        if not os.path.exists(result_path):
            raise Exception("Try-on result not generated")
            
        im = Image.open(result_path)
        width, height = im.size
        
        # Adjust the cropping to show more of the outfit
        left = width / 4
        top = height / 4
        right = width * 3/4
        bottom = height
        
        im1 = im.crop((left, top, right, bottom))
        newsize = (600, 800)  # Increased height to show more of the outfit
        im1 = im1.resize(newsize)
        im1.save("data.jpg")
        
        # Get product details
        product_info = product_details.get(selection, {})
        
        return jsonify({
            'image': "data.jpg",
            'product': product_info
        })
    except Exception as e:
        print(f"Error in final_img: {str(e)}")
        return jsonify({
            'error': str(e),
            'image': "temp.jpg"
        })
    
    return Response(gen_stored("data.jpg"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_fashion_advice', methods=['POST'])
def get_fashion_advice():
    data = request.json
    body_type = data.get('body_type')
    skin_tone = data.get('skin_tone')
    personal_style = data.get('personal_style')
    
    tips = get_fashion_tips(body_type, skin_tone, personal_style)
    return jsonify({'tips': tips})

@app.route('/get_outfit_recommendation', methods=['POST'])
def get_outfit_recommendation():
    data = request.json
    occasion = data.get('occasion')
    style_preference = data.get('style_preference')
    weather = data.get('weather')
    
    suggestions = get_outfit_suggestions(occasion, style_preference, weather)
    return jsonify({'suggestions': suggestions})

@app.route('/analyze_outfit', methods=['POST'])
def analyze_outfit():
    data = request.json
    outfit_description = data.get('outfit_description')
    
    analysis = analyze_current_outfit(outfit_description)
    return jsonify({'analysis': analysis})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)