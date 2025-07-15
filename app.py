import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import numpy



st.set_page_config(page_title= "Europe Area Detector", layout= "wide")

st.title("Statellite Area Detector (Europe)")
st.write("Select an area on European map, capture the image, and let the model identify")

import folium
from streamlit_folium import st_folium




import streamlit as st
import pyautogui
from PIL import Image
import os

st.write("🗺️ Crop Map Screenshot")



# 3. Crop and Show the Image
if st.button("✂️ Crop and Save Map Area"):
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot.png")
        st.image("screenshot.png", caption="Full Screenshot")

        image = Image.open("screenshot.png")
        crop_box = (850, 420, 1200+256, 420+256)
        cropped = image.crop(crop_box)
        cropped.save("screenshot.png")
        st.image("screenshot.png", caption="🗺️ Cropped Map Area")
        st.success("Cropped image saved successfully!")

        with open("cropped_map.png", "rb") as file:
            st.download_button("📥 Download Cropped Image", file, file_name="map_crop.png")

    except Exception as e:
        st.error(f"Error cropping image: {e}")



#=========================================================




# Create a map
# m = folium.Map(location=[20, 0], zoom_start=2, tiles="Stamen Terrain")  # or tiles="OpenStreetMap" or use satellite


m = m = folium.Map(
    location=[51.5072, 0.1276],  # Center of the world map
    zoom_start=4,
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # Satellite tiles
    attr='Google Satellite'
)
folium.TileLayer("OpenStreetMap",crs = "EPSG4326").add_to(m)
folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="Openstreet").add_to(m)
folium.LayerControl().add_to(m)


# zoom levle saving=======================


# Let user draw rectangle on map
from folium.plugins import Draw
draw = Draw(export=True)
draw.add_to(m)

# Display map
st_data = st_folium(m, width=1350, height=500)

# Display current zoom
if st_data:
    zoom_level = st_data.get("zoom")
    if zoom_level is not None:
        st.success(f"🔍 Current Zoom Level: {zoom_level}")





# Model Loading
import torch
import torch.nn as nn
from torchvision import transforms, models

model = models.resnet34(pretrained = False)
model.fc = nn.Linear(model.fc.in_features,10)

model.load_state_dict(torch.load('model_eurosat.pth'))

model= model.to("cpu")
model.eval()



transform =transforms.Compose([
    transforms.Resize(234), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

input_image = Image.open("screenshot.png").convert("RGB")
input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0)

output = model(input_tensor)

prob = torch.softmax(output, dim=1)

value, label = torch.max(prob, dim=1)

value = value.item()
label = label.item()
# value = value.numpy()
# label = label.numpy()

label_name= {0: 'AnnualCrop',
             1: 'Forest',
             2: 'HerbaceousVegetation', 3: 'Highway', 4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential',
             8: 'River', 9: 'SeaLake'}

st.image(input_image)
display = input_tensor.squeeze(0)
display = torch.clamp(display, 0,1)
display = display.numpy()
st.image(numpy.transpose(display, [1,2,0]))
st.write(f"Probability {value}")
st.write(f'{label_name[label]}')

