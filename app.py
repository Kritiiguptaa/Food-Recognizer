# app.py — Fixed TTA at 7 (No Slider)

from pathlib import Path
import io
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import streamlit as st
from torchvision import transforms
from nutrition import nutrition_for_dish  

st.set_page_config(page_title="Indian Food Recognizer", page_icon="🍽️", layout="wide")

ROOT = Path(__file__).resolve().parent
CUISINE_ROOT = ROOT / "cuisines"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm(s):
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum() or ch.isspace())

@st.cache_resource
def load_classes(path):
    return [line.strip() for line in open(path, "r", encoding="utf-8")]

@st.cache_resource
def load_model(path, num_classes):
    path = str(path)
    if "b3" in path.lower():
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m

@st.cache_data
def load_excel(path):
    df = pd.read_excel(path)
    df["Recipe Name"] = df["Recipe Name"].astype(str)
    return df

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(img, model, classes, tta=3):
    x = TFM(img.convert("RGB")).unsqueeze(0).to(device)
    logits = sum(model(torch.flip(x, dims=[3]) if i % 2 else x) for i in range(tta))
    probs = (logits / tta).softmax(1).squeeze(0)
    p, idx = probs.topk(5)
    return [(classes[i], float(pp)) for pp, i in zip(p, idx)]

def lookup_recipe(name, df):
    key = _norm(name)
  
    for r in df["Recipe Name"]:
        if _norm(r) == key:
            return df[df["Recipe Name"] == r].iloc[0].to_dict()
 
    for r in df["Recipe Name"]:
        if key in _norm(r):
            return df[df["Recipe Name"] == r].iloc[0].to_dict()
    return None


def load_cuisine(folder):
    path = CUISINE_ROOT / folder
    classes = load_classes(path / "models" / "classes.txt")
    num_c = len(classes)
    if "South" in folder:
        model_path = path / "models" / "best_b3.pth"
    else:
        model_path = path / "models" / "best_efficientnet_b0.pth"

    return {
        "model": load_model(model_path, num_c),
        "classes": classes,
        "recipes": load_excel(path / "recipes.xlsx")
    }

ALL = {
    "Punjabi": load_cuisine("Punjabi Cuisines"),
    "South Indian": load_cuisine("South Indian Cuisines")
}


st.title("🍽️ Indian Food Recognizer")
st.markdown("Upload a photo of Indian food to identify the dish, get the recipe, and see nutrition info.")

col1, col2 = st.columns(2)
uploaded = col1.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera = col2.camera_input("Take Photo")


tta = 7  

if uploaded or camera:
    file = camera if camera else uploaded
    img = Image.open(io.BytesIO(file.getvalue()))
    
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img, use_container_width=True, caption="Your Food")

    
    res = {}
    
    with st.spinner("Analyzing flavors (high accuracy mode)..."):
        for cname, C in ALL.items():
            res[cname] = predict(img, C["model"], C["classes"], tta)

    best_cuisine = max(res, key=lambda c: res[c][0][1])
    top5 = res[best_cuisine]
    best_dish = top5[0][0]

   
    st.divider()
    
   
    st.markdown(f"<h2 style='text-align: center;'>👉 It looks like: <b>{best_dish}</b></h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>Cuisine: {best_cuisine}</p>", unsafe_allow_html=True)

   
    with st.expander("See other possibilities"):
        st.table(pd.DataFrame(top5, columns=["Dish Name", "Confidence Score"]))

    
    C = ALL[best_cuisine]
    recipe = lookup_recipe(best_dish, C["recipes"])

    if recipe:
        st.markdown("### 📘 Recipe Card")
        
       
        m1, m2, m3 = st.columns(3)
        time_val = recipe.get("Total Time", recipe.get("Total TimeInMinutes", "N/A"))
        m1.metric("⏱️ Total Time", f"{time_val} min" if str(time_val).isdigit() else str(time_val))
        m2.metric("🥘 Course", recipe.get("Course", "Main Dish"))
        m3.metric("🌶️ Diet", recipe.get("Diet", "Vegetarian"))

        st.markdown("") # Spacer

        # 2. Ingredients & Instructions Columns
        col_ing, col_inst = st.columns([1, 1.5]) # Left narrow, Right wide

        with col_ing:
            st.info("**🛒 Ingredients**")
            # Split comma-separated ingredients into a list
            raw_ing = str(recipe.get("Ingredients", ""))
            ingredients = [x.strip() for x in raw_ing.split(',') if x.strip()]
            
            for ing in ingredients:
                st.markdown(f"• {ing}")

        with col_inst:
            st.success("**👩‍🍳 Instructions**")
            # Split paragraph text into sentences based on '.'
            raw_inst = str(recipe.get("Instructions", ""))
            steps = [s.strip() for s in raw_inst.split('.') if len(s.strip()) > 5]
            
            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}.")

        # 3. Nutrition Section
        st.divider()
        st.subheader("🥗 Nutrition Facts (Estimated)")
        
        cleaned = recipe.get("Cleaned-Ingredients", "") or recipe.get("Ingredients", "")
        ing_list = [i.strip() for i in str(cleaned).split(",") if i.strip()]

        with st.spinner("Fetching USDA data..."):
            nut, source = nutrition_for_dish(best_dish, ing_list)

        if nut:
            st.caption(f"Source: {source}")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("🔥 Calories", f"{nut['calories']} kcal")
            n2.metric("💪 Protein", f"{nut['protein']} g")
            n3.metric("🥑 Fat", f"{nut['fat']} g")
            n4.metric("🍬 Sugar", f"{nut['sugar']} g")
        else:
            st.warning("Could not calculate nutrition for this dish.")

        # Link
        url = recipe.get("URL") or recipe.get("Link to recipe")
        if url:
            st.markdown(f"[🔗 **View Original Recipe on Archana's Kitchen**]({url})")

    else:
        st.error(f"Recipe details for '{best_dish}' not found in the database.")