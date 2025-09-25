# Import the necessary modules
import requests
import json
import os
import base64
from PIL import Image
import io
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Stability API
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
STABILITY_API_HOST = 'https://api.stability.ai'

def generate_fashion_image(prompt, style="digital-art", size=512):
    """Generate fashion images using Stability AI"""
    try:
        url = f"{STABILITY_API_HOST}/v1/generation/stable-diffusion-v1-5/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        body = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": size,
            "width": size,
            "samples": 1,
            "sampler": "K_DPM_2_ANCESTRAL",
            "style_preset": style
        }
        
        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code == 200:
            data = response.json()
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            image = Image.open(io.BytesIO(image_data))
            return image
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def get_style_recommendation(outfit_type, user_preferences=None):
    """Get AI-powered style recommendations and generate example outfit"""
    try:
        # Generate prompt for the outfit
        style_prompt = f"Fashion photograph of a {outfit_type}, professional studio lighting, high-end fashion"
        if user_preferences:
            style_prompt += f", {user_preferences}"
        
        # Generate image using Stability AI
        generated_image = generate_fashion_image(style_prompt)
        
        if generated_image:
            # Save the generated image to a temporary file
            temp_path = "static/images/generated_outfit.jpg"
            generated_image.save(temp_path)
            return {
                "success": True,
                "image_path": temp_path,
                "description": f"Generated outfit suggestion for {outfit_type}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate outfit image"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating style recommendation: {str(e)}"
        }

def analyze_outfit_compatibility(clothing_item, user_image):
    """Analyze if a clothing item would look good on the user and generate preview"""
    try:
        # Generate a prompt for outfit compatibility
        compatibility_prompt = f"Fashion photograph showing {clothing_item} styled on a model, professional fashion photography"
        
        # Generate visualization using Stability AI
        preview_image = generate_fashion_image(compatibility_prompt)
        
        if preview_image:
            # Save the preview image
            preview_path = "static/images/outfit_preview.jpg"
            preview_image.save(preview_path)
            return {
                "success": True,
                "image_path": preview_path,
                "message": f"Generated preview of how {clothing_item} might look"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate outfit preview"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing outfit compatibility: {str(e)}"
        }

def get_outfit_suggestions(occasion, style_preference, weather=None):
    """Get AI-powered outfit suggestions with generated images"""
    try:
        # Create prompt based on occasion and style
        prompt = f"Fashion photograph of a complete outfit for {occasion}, {style_preference} style"
        if weather:
            prompt += f", suitable for {weather} weather"
            
        # Generate outfit visualization
        outfit_image = generate_fashion_image(prompt)
        
        if outfit_image:
            # Save the generated outfit
            outfit_path = "static/images/suggested_outfit.jpg"
            outfit_image.save(outfit_path)
            return {
                "success": True,
                "image_path": outfit_path,
                "description": f"Generated outfit suggestion for {occasion}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate outfit suggestion"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating outfit suggestion: {str(e)}"
        }

def analyze_current_outfit(image_path):
    """Analyze current outfit and provide enhancement suggestions"""
    try:
        # Load the current outfit image
        current_image = Image.open(image_path)
        
        # Generate an enhanced version
        enhance_prompt = "Enhanced fashion photograph, professional styling, high-end fashion photography"
        enhanced_image = generate_fashion_image(enhance_prompt)
        
        if enhanced_image:
            # Save the enhanced suggestion
            enhanced_path = "static/images/enhanced_outfit.jpg"
            enhanced_image.save(enhanced_path)
            return {
                "success": True,
                "image_path": enhanced_path,
                "message": "Generated enhanced outfit suggestion"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate enhancement suggestion"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing current outfit: {str(e)}"
        }

def get_fashion_tips(style_type):
    """Get fashion tips with example images"""
    try:
        # Generate visual example for the fashion tip
        tip_prompt = f"Fashion photograph demonstrating {style_type} style tips, professional fashion photography"
        tip_image = generate_fashion_image(tip_prompt)
        
        if tip_image:
            # Save the tip visualization
            tip_path = "static/images/fashion_tip.jpg"
            tip_image.save(tip_path)
            return {
                "success": True,
                "image_path": tip_path,
                "tip": f"Visual fashion tip for {style_type} style"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate fashion tip visualization"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating fashion tip: {str(e)}"
        }
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fashion AI expert who creates personalized outfit recommendations."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate outfit suggestions. Error: {str(e)}"

def get_fashion_tips(body_type=None, skin_tone=None, personal_style=None):
    """Get personalized fashion tips based on user characteristics"""
    prompt = "Provide fashion tips"
    if body_type:
        prompt += f" for {body_type} body type"
    if skin_tone:
        prompt += f" and {skin_tone} skin tone"
    if personal_style:
        prompt += f" considering {personal_style} personal style"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fashion AI expert who provides personalized fashion advice."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate fashion tips. Error: {str(e)}"

def analyze_current_outfit(image_description):
    """Analyze current outfit and provide feedback"""
    prompt = f"Analyze this outfit and provide constructive feedback: {image_description}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fashion AI expert who provides detailed outfit analysis and feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't analyze the outfit. Error: {str(e)}"