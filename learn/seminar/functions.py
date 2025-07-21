import streamlit as st
from PIL import Image
from PIL import ImageDraw
import base64

# ---------------------Функция отрисовки лиинй для bounding-box-а---------------------
def draw_boxes(image, bounds, color='yellow', width=2):
    image = Image.open(image) # дописала, чтобы загруженное img перевести в "путь" к нему (str)
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


# ---------------------Функция замены фона в стримлите---------------------
def set_background(main_bg):
    '''
    A function to unpack an image from root folder and set as bg. 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
